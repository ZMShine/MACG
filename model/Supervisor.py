import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from lib import utils
from model.model import MACGModel
from model.loss import masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mae_loss_single
import pandas as pd
import os
import time
import math
from line_profiler import profile
import torch.autograd.profiler as profiler
import importlib
from scipy.spatial.distance import cdist
import fastdtw
from scipy.spatial.distance import euclidean
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MACGSupervisor:
    def __init__(self, temperature, **kwargs):
        # Read paramesters:
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._test_kwargs = kwargs.get('test')
        self.temperature = float(temperature)
        self.opt = self._train_kwargs.get('optimizer')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.train_batchsize = int(self._data_kwargs.get('batch_size', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # logging
        experiment_id = str(int(time.time()))
        self._log_dir = self._get_log_dir(kwargs)
        log_file = f'info_{experiment_id}.log'
        self._writer = SummaryWriter('runs/' + self._log_dir)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, log_file, level=log_level)

        # processed data loading
        self._data = utils.load_dataset(**self._data_kwargs)  # 加载划分好的数据集
        self.standard_scaler = self._data['scaler']  # 数据集标准化

        # 不同数据集读取
        df = pd.read_hdf('./data/metr-la.h5')
        # df = pd.read_hdf('./data/pems-bay.h5')
        # df = pd.read_csv('./data/FeatureGroup/feature3.csv')

        # 处理空值，上下均值填充空值
        df = df.fillna(df.interpolate())

        # 选取与训练集相同的数据量作为特征训练KNN graph
        num_samples = df.shape[0]  # 获得df的行数,[1]是获取列数
        num_train = round(num_samples * 0.7) - 16  # 对行数*0.7再取四舍五入
        df = df[:num_train].values  # 取前70%列的数据

        # 初始化特征矩阵
        df = torch.Tensor(df).to(device)
        # 训练数据（特征数据）标准化
        scaler = utils.StandardScaler(mean=df.mean(), std=df.std())  # 计算均值标准差
        self._train_feas = scaler.transform(df)  # 训练特征标准化

        # setup model
        MACG_model = MACGModel(self.temperature, self._logger, **self._model_kwargs)
        self.MACG_model = MACG_model.to(device)
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 1)
        if self._epoch_num > 0:
            self.load_model()

        self.batches_seen = 0

    @staticmethod
    def _get_log_dir(kwargs):
        # offline training summary
        base_lr = kwargs['train'].get('base_lr')
        max_diffusion_step = kwargs['model'].get('max_diffusion_step')
        num_rnn_layers = kwargs['model'].get('num_rnn_layers')
        rnn_units = kwargs['model'].get('rnn_units')
        structure = '-'.join(['%d' % rnn_units for _ in range(num_rnn_layers)])

        # online testing summary
        online_rate_nodrift = kwargs['test'].get('online_rate_nodrift')
        online_rate_drift = kwargs['test'].get('online_rate_drift')
        match_size = kwargs['test'].get('match_size')
        val_scale = kwargs['test'].get('val_scale')

        # log initialization
        run_id = ('MACG_baselr_%g_%d_%s_ollr_%g_%g_%d_%g_%s/'
                  % (base_lr, max_diffusion_step, structure, online_rate_nodrift, online_rate_drift, match_size, val_scale,
                     time.strftime('%m%d%H%M%S')))
        base_dir = kwargs.get('log_dir')
        log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        return log_dir

    # model save
    def save_model(self, epoch):
        if not os.path.exists('models/'): os.makedirs('models/')
        torch.save({'model': self.MACG_model, 'graph_data': self.sampledgraph}, 'models/epo%d.pth' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.pth' % epoch

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def load_model(self):
        assert os.path.exists('models/epo%d.pth' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.pth' % self._epoch_num)
        self.MACG_model = checkpoint['model']
        self.sampledgraph = checkpoint['graph_data']
        self._logger.info("Loaded model at {}".format(self._epoch_num))
        self._setup_graph()

    def _setup_graph(self):
        # record performance on validation set
        self.staticloss, self.staticloss_Step1 = self.evaluate(dataset='val')

        # Testing under offline
        self._data = utils.load_dataset(dataset_dir = 'data/Processed Data', batch_size = 64, val_batch_size = 64, test_batch_size=64)
        test_loss, test_mape, test_rmse = self.evaluate_test_offline(dataset='test', batches_seen=self._epoch_num * self._data['train_loader'].num_batch)
        message = 'The overall offline performance: test_mae: {:.4f}, test_mape: {:.4f}, test_rmse: {:.4f}'.format(test_loss, test_mape, test_rmse)
        self._logger.info(message)

        # Testing under onfline
        self._data = utils.load_dataset(**self._data_kwargs)
        test_loss, test_mape, test_rmse = self.evaluate_test(dataset='test', batches_seen=self._epoch_num * self._data['train_loader'].num_batch)
        message = 'The online performance: test_mae: {:.4f}, test_mape: {:.4f}, test_rmse: {:.4f}'.format(test_loss, test_mape, test_rmse)
        self._logger.info(message)

        self._writer.close()
        self._logger.handlers.clear()
        return

    # Test on offline validation set
    def evaluate(self, dataset='val'):
        with torch.no_grad():
            self.MACG_model = self.MACG_model.eval()
            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            step1 = []
            for batch_idx, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.MACG_model(x, self.sampledgraph)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)
                step1.append(masked_mae_loss(y_pred[0:1], y_true[0:1]).item())
            mean_loss = np.mean(losses)
            mean_step1 = np.mean(step1)
        return mean_loss, mean_step1

    def evaluate_test_offline(self, dataset='test', batches_seen=0):
        test_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
        losses = []
        mapes = []
        rmses = []

        l_3 = []
        m_3 = []
        r_3 = []
        l_6 = []
        m_6 = []
        r_6 = []
        l_12 = []
        m_12 = []
        r_12 = []

        # 测试集按照小batch进行测试模拟数据流：
        for batch_idx, (x, y) in enumerate(test_iterator):
            # 准备测试数据
            x, y = self._prepare_data(x, y)

            # 还原每个batch的预测真值
            y_true = self.standard_scaler.inverse_transform(y)

            # 加载预测模型：
            with torch.no_grad():
                self.MACG_model = self.MACG_model.eval()
                output = self.MACG_model(x, self.sampledgraph)
                loss = self._compute_loss(y, output)
                y_pred = self.standard_scaler.inverse_transform(output)

                # 计算模型精度
                mapes.append(masked_mape_loss(y_pred, y_true).item())
                rmses.append(masked_rmse_loss(y_pred, y_true).item())
                losses.append(loss.item())

                l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
                m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
                r_3.append(masked_rmse_loss(y_pred[2:3], y_true[2:3]).item())
                l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
                m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
                r_6.append(masked_rmse_loss(y_pred[5:6], y_true[5:6]).item())
                l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
                m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
                r_12.append(masked_rmse_loss(y_pred[11:12], y_true[11:12]).item())

        mean_loss = np.mean(losses)
        mean_mape = np.mean(mapes)
        mean_rmse = np.mean(rmses)

        # Followed the DCRNN PyTorch Implementation
        message = 'Test_offline: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_loss, mean_mape, mean_rmse)
        self._logger.info(message)

        message = 'Horizon_offline 3h: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_3), np.mean(m_3),
                                                                                       np.mean(r_3))
        self._logger.info(message)
        message = 'Horizon_offline 6h: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_6), np.mean(m_6),
                                                                                       np.mean(r_6))
        self._logger.info(message)
        message = 'Horizon_offline 12h: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_12), np.mean(m_12),
                                                                                        np.mean(r_12))
        self._logger.info(message)
        self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

        return mean_loss, mean_mape, mean_rmse

    # Online testing stage
    # @profile
    def evaluate_test(self, dataset='test', batches_seen=0):

        # Parameter Initilization
        online_rate_nodrift = float(self._test_kwargs.get('online_rate_nodrift'))
        online_rate_drift = float(self._test_kwargs.get('online_rate_drift'))
        online_epoch_nodrift = int(self._test_kwargs.get('online_epoch_nodrift'))
        online_epoch_drift = int(self._test_kwargs.get('online_epoch_drift'))
        match_size = int(self._test_kwargs.get('match_size'))
        val_scale = float(self._test_kwargs.get('val_scale'))
        test_iterator = self._data['{}_loader'.format(dataset)].get_iterator()

        losses = []
        mapes = []
        rmses = []

        # 增加多个预测点
        l_1 = []
        m_1 = []
        r_1 = []
        l_2 = []
        m_2 = []
        r_2 = []

        l_3 = []
        m_3 = []
        r_3 = []
        l_6 = []
        m_6 = []
        r_6 = []
        l_12 = []
        m_12 = []
        r_12 = []

        # 设置特征存储初始状态
        Feature_store = self._train_feas

        # 初始化新数据滑动窗口
        X_SampleStore = torch.empty(0, 2 * self.num_nodes).to(device)
        Y_SampleStore = torch.empty(0, self.num_nodes).to(device)

        # 设置数据窗口读取的flag
        t = 0

        # record testing time
        Testing_start_time = time.time()

        # Draw information for concept drift adaptation
        Loss_draw = []
        Dy_loss_record = []
        Dy_loss_record.append(self.staticloss_Step1)

        Data_history_X = []
        Data_history_Y = []
        train_iterator = self._data['train_loader'].get_iterator()
        for batch_idx, (x, y) in enumerate(train_iterator):
            x, y = self._prepare_data_history(x, y)
            Data_history_X.append(x)
            Data_history_Y.append(y)
        Data_history_X = torch.cat(Data_history_X, dim=1)
        Data_history_Y = torch.cat(Data_history_Y, dim=1)
        X_new_SampleStore = torch.empty(0, self.num_nodes, self.input_dim).to(device)

        self.MACG_model.alpha = nn.Parameter(torch.tensor(0.001))

        self.MACG_model_temp = None


        # 测试集按照小batch进行测试模拟数据流：
        for batch_idx, (x, y) in enumerate(test_iterator):
            print("current batch:", batch_idx)

            # Update feature
            x_feature, y_feature = self._prepare_feature(x, y)
            y_feature = y_feature[0, :, :]
            if batch_idx == 0:
                # t<11, feature 拼接 x
                Feature_store = torch.cat((Feature_store, x_feature.reshape(self.horizon, self.num_nodes)))

            # 准备测试数据
            x_new, y_new = self._prepare_data_history(x, y)
            x, y = self._prepare_data(x, y)

            # New Sample store
            # 样本逐个到来，所以只拼接第一个值
            X_SampleStore = torch.cat((X_SampleStore, x[0]), 0)
            Y_SampleStore = torch.cat((Y_SampleStore, y[0]), 0)

            X_new_SampleStore = torch.cat((X_new_SampleStore, x_new[0]), 0)

            # 还原每个batch的预测真值
            y_true = self.standard_scaler.inverse_transform(y)

            if batch_idx < 11:
                # 加载预测模型：
                with torch.no_grad():
                    self.MACG_model = self.MACG_model.eval()
                    output = self.MACG_model(x, self.sampledgraph)
                    loss = self._compute_loss(y, output)
                    y_pred = self.standard_scaler.inverse_transform(output)

                    # 计算模型精度
                    mapes.append(masked_mape_loss(y_pred, y_true).item())
                    rmses.append(masked_rmse_loss(y_pred, y_true).item())
                    losses.append(loss.item())

                    # 计算多个步长精度
                    l_1.append(masked_mae_loss(y_pred[0:1], y_true[0:1]).item())
                    print("Step1mae:", masked_mae_loss(y_pred[0:1], y_true[0:1]).item())
                    m_1.append(masked_mape_loss(y_pred[0:1], y_true[0:1]).item())
                    r_1.append(masked_rmse_loss(y_pred[0:1], y_true[0:1]).item())
                    l_2.append(masked_mae_loss(y_pred[1:2], y_true[1:2]).item())
                    m_2.append(masked_mape_loss(y_pred[1:2], y_true[1:2]).item())
                    r_2.append(masked_rmse_loss(y_pred[1:2], y_true[1:2]).item())
                    l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
                    m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
                    r_3.append(masked_rmse_loss(y_pred[2:3], y_true[2:3]).item())
                    l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
                    m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
                    r_6.append(masked_rmse_loss(y_pred[5:6], y_true[5:6]).item())
                    l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
                    m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
                    r_12.append(masked_rmse_loss(y_pred[11:12], y_true[11:12]).item())

                # 特征更新窗口拼接
                Feature_store = torch.cat((Feature_store, y_feature))

            else:
                with torch.no_grad():

                    self.MACG_model = self.MACG_model.eval()
                    output = self.MACG_model(x, self.sampledgraph)
                    y_pred = self.standard_scaler.inverse_transform(output)

                    loss = self._compute_loss(y, output)

                    # 计算整个测试集精度
                    mapes.append(masked_mape_loss(y_pred, y_true).item())
                    rmses.append(masked_rmse_loss(y_pred, y_true).item())
                    losses.append(loss.item())

                    # Record Loss for test adaptation speed
                    # Loss_draw.append(loss.item())
                    # if batch_idx < 1000:
                    #     Loss_draw.append(loss.item())
                    # else:
                    #     Loss_draw = pd.DataFrame({'Loss_recording': Loss_draw})
                    #     Loss_draw.to_csv("Loss_DyOLr.csv", index=False, sep=',')
                    #     print("The loss hase been recorded.")
                    #     break

                    # Followed the DCRNN TensorFlow Implementation
                    l_1.append(masked_mae_loss(y_pred[0:1], y_true[0:1]).item())
                    print("Step1mae:", masked_mae_loss(y_pred[0:1], y_true[0:1]).item())
                    Dy_loss_record.append(masked_mae_loss(y_pred[0:1], y_true[0:1]).item())
                    m_1.append(masked_mape_loss(y_pred[0:1], y_true[0:1]).item())
                    r_1.append(masked_rmse_loss(y_pred[0:1], y_true[0:1]).item())
                    l_2.append(masked_mae_loss(y_pred[1:2], y_true[1:2]).item())
                    m_2.append(masked_mape_loss(y_pred[1:2], y_true[1:2]).item())
                    r_2.append(masked_rmse_loss(y_pred[1:2], y_true[1:2]).item())
                    l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
                    m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
                    r_3.append(masked_rmse_loss(y_pred[2:3], y_true[2:3]).item())
                    l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
                    m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
                    r_6.append(masked_rmse_loss(y_pred[5:6], y_true[5:6]).item())
                    l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
                    m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
                    r_12.append(masked_rmse_loss(y_pred[11:12], y_true[11:12]).item())

                # 截取窗口数据作为online更新的samples
                X_update = X_SampleStore[t:t + self.horizon, :].reshape(self.horizon, 1, 2 * self.num_nodes)
                Y_update = Y_SampleStore[t:t + self.horizon, :].reshape(self.horizon, 1, self.num_nodes)

                X_new_update = X_new_SampleStore[t:t + self.horizon, :].reshape(self.horizon, 1, self.num_nodes, self.input_dim)

                Data_history_X = torch.cat((Data_history_X, X_new_update), dim=1)
                Data_history_Y = torch.cat((Data_history_Y, Y_update), dim=1)

                t = t + 1

                stop_update = self.check_stop_update(self.standard_scaler.inverse_transform(X_update),
                                                     self.standard_scaler.inverse_transform(Y_update), threshold=0.3)
                if not stop_update:

                    # Data and feature preparation for adaptation
                    # Update feature store
                    Feature_store = torch.cat((Feature_store, y_feature))
                    feature_window_temp = Feature_store[-self._train_feas.size(0):, :]


                    # region Drift Adaptation Version 1 (Only Drift Detection and Updating)
                    if np.mean(Dy_loss_record) < masked_mae_loss(y_pred[0:1], y_true[0:1]).item():
                        self._logger.info('Drift occurs...')

                        top_k_indices = self.find_matching_windows(Y_update, Data_history_X, top_k=match_size)
                        matched_X, matched_Y = self.extract_matched_data(Data_history_X, Data_history_Y, top_k_indices)

                        matched_X = matched_X.reshape(self.seq_len, top_k_indices.size(0),
                                                      self.num_nodes * self.input_dim)
                        X_update_ad = torch.cat((X_update, matched_X), dim=1)
                        Y_update_ad = torch.cat((Y_update, matched_Y), dim=1)

                        self.MACG_model_temp = copy.deepcopy(self.MACG_model)


                        for epoch_online in range(online_epoch_drift):
                            optimizer_OL = torch.optim.Adam(self.MACG_model.parameters(), lr=online_rate_drift, eps=1.0e-3)
                            self.MACG_model = self.MACG_model.train()
                            optimizer_OL.zero_grad()
                            Feature_adj = self.MACG_model.FeatureExtraction(self._train_feas, temp=self.temperature)
                            self.sampledgraph = Feature_adj
                            Feature_adj_online = self.MACG_model.FeatureExtraction(feature_window_temp, temp=self.temperature)
                            dy_graph = Feature_adj_online * self.MACG_model.alpha + self.sampledgraph * (1 - self.MACG_model.alpha)
                            output = self.MACG_model(X_update_ad, dy_graph, Y_update_ad, batches_seen)
                            loss_OL = self._compute_loss(Y_update_ad, output)
                            self._logger.debug(loss_OL.item())
                            batches_seen += 1
                            loss_OL.backward()
                            torch.nn.utils.clip_grad_norm_(self.MACG_model.parameters(), self.max_grad_norm)
                            optimizer_OL.step()

                        for epoch_online in range(online_epoch_drift):
                            optimizer_OL = torch.optim.Adam(self.MACG_model_temp.parameters(), lr=online_rate_drift, eps=1.0e-3)
                            self.MACG_model_temp = self.MACG_model_temp.train()
                            optimizer_OL.zero_grad()
                            Feature_adj = self.MACG_model_temp.FeatureExtraction(self._train_feas, temp=self.temperature)
                            self.sampledgraph = Feature_adj
                            Feature_adj_online = self.MACG_model_temp.FeatureExtraction(feature_window_temp,temp=self.temperature)
                            dy_graph = Feature_adj_online * self.MACG_model_temp.alpha + self.sampledgraph * (1 - self.MACG_model_temp.alpha)
                            output = self.MACG_model_temp(X_update, dy_graph, Y_update, batches_seen)
                            loss_OL = self._compute_loss(Y_update, output)
                            self._logger.debug(loss_OL.item())
                            batches_seen += 1
                            loss_OL.backward()
                            torch.nn.utils.clip_grad_norm_(self.MACG_model_temp.parameters(), self.max_grad_norm)
                            optimizer_OL.step()

                        validation_ol = self.validation_olline(self.MACG_model, self.MACG_model_temp, val_scale)
                        if validation_ol == False:
                            self.MACG_model.load_state_dict(self.MACG_model_temp.state_dict())

                    else:
                        for epoch_online in range(online_epoch_nodrift):
                            optimizer_OL = torch.optim.Adam(self.MACG_model.parameters(), lr=online_rate_nodrift,
                                                            eps=1.0e-3)
                            self.MACG_model = self.MACG_model.train()
                            optimizer_OL.zero_grad()
                            Feature_adj = self.MACG_model.FeatureExtraction(self._train_feas, temp=self.temperature)
                            self.sampledgraph = Feature_adj
                            output = self.MACG_model(X_update, self.sampledgraph, Y_update, batches_seen)
                            loss_OL = self._compute_loss(Y_update, output)
                            self._logger.debug(loss_OL.item())
                            batches_seen += 1
                            loss_OL.backward()
                            torch.nn.utils.clip_grad_norm_(self.MACG_model.parameters(), self.max_grad_norm)
                            optimizer_OL.step()
                    # endregion

                else:
                    print("The GNN Update paused due to outliers.")

        Testing_end_time = time.time()
        self._logger.info('The average testing time of each sample is: {:.4f}'.format(
            (Testing_end_time - Testing_start_time) / (self._data['test_loader'].num_batch)))
        mean_loss = np.mean(losses)
        mean_mape = np.mean(mapes)
        mean_rmse = np.mean(rmses)

        # Followed the DCRNN PyTorch Implementation
        message = 'Test: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_loss, mean_mape, mean_rmse)
        self._logger.info(message)

        # Followed the DCRNN TensorFlow Implementation
        message = 'Horizon 1h: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_1), np.mean(m_1), np.mean(r_1))
        self._logger.info(message)
        message = 'Horizon 2h: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_2), np.mean(m_2), np.mean(r_2))
        self._logger.info(message)
        message = 'Horizon 3h: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_3), np.mean(m_3), np.mean(r_3))
        self._logger.info(message)
        message = 'Horizon 6h: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_6), np.mean(m_6), np.mean(r_6))
        self._logger.info(message)
        message = 'Horizon 12h: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_12), np.mean(m_12),
                                                                                np.mean(r_12))
        self._logger.info(message)

        self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)
        return mean_loss, mean_mape, mean_rmse

    def _train(self, base_lr, steps, patience, epochs, lr_decay_ratio, epsilon=1e-3, **kwargs):
        min_val_loss = float('inf')
        wait = 0
        if self.opt == 'adam':
            optimizer = torch.optim.Adam(self.MACG_model.parameters(), lr=base_lr, eps=epsilon)
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.MACG_model.parameters(), lr=base_lr)
        else:
            optimizer = torch.optim.Adam(self.MACG_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=float(lr_decay_ratio))

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        Training_time = []
        Validation_time = []

        for epoch_num in range(self._epoch_num, epochs):
            print("Num of epoch:", epoch_num)
            self.MACG_model = self.MACG_model.train()
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []

            # record time
            start_time = time.time()

            for batch_idx, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()
                x, y = self._prepare_data(x, y)

                # 生成sampled graph
                Feature_adj = self.MACG_model.FeatureExtraction(self._train_feas, temp=self.temperature)
                self.sampledgraph = Feature_adj

                output = self.MACG_model(x, self.sampledgraph, y, batches_seen)

                if batches_seen == 0:  # batches_seen = num_batches * self._epoch_num
                    if self.opt == 'adam':
                        optimizer = torch.optim.Adam(self.MACG_model.parameters(), lr=base_lr, eps=epsilon)
                    elif self.opt == 'sgd':
                        optimizer = torch.optim.SGD(self.MACG_model.parameters(), lr=base_lr)
                    else:
                        optimizer = torch.optim.Adam(self.MACG_model.parameters(), lr=base_lr, eps=epsilon)

                self.MACG_model.to(device)
                loss = self._compute_loss(y, output)
                losses.append((loss.item()))
                self._logger.debug(loss.item())
                batches_seen += 1
                loss.backward()
                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.MACG_model.parameters(), self.max_grad_norm)
                optimizer.step()

            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")
            # record training time each epoch
            end_time = time.time()
            Training_time.append(end_time - start_time)

            # 测试每个epoch训练后的model性能
            val_loss, val_loss_step1 = self.evaluate(dataset='val')
            # record validation time
            end_time2 = time.time()
            Validation_time.append(end_time2 - end_time)
            self._writer.add_scalar('training loss', np.mean(losses), batches_seen)
            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}'.format(epoch_num, epochs, batches_seen,
                                                                                     np.mean(losses), val_loss)
            self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                Best_epoch = epoch_num
                # 记录训练时的最小val_loss
                min_val_loss = val_loss
                # 保存current best epoch训练后的GNN
                model_file_name = self.save_model(epoch_num)
                self._logger.info('saving to {}'.format(model_file_name))
                print("model has been saved.")

            elif val_loss >= min_val_loss:
                wait += 1
                print("wait number: ", wait)
                # early stop
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    self._logger.info(
                        'The average training time of each epoch is: {:.4f}'.format(np.mean(Training_time)))
                    self._logger.info(
                        'The average validation time of each epoch is: {:.4f}'.format(np.mean(Validation_time)))
                    self._logger.info('The Best_epoch in the training stage is Epoch: %d' % Best_epoch)
                    self._epoch_num = Best_epoch
                    self.load_model()
                    break
        return

    # @profile
    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x, y

    # @profile
    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    # @profile
    def _get_x_y_in_correct_dims(self, x, y):
        """batch_size
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.reshape(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].reshape(self.horizon, batch_size,
                                             self.num_nodes * self.output_dim)

        return x, y

    def _prepare_data_history(self, x, y):
        x, y = self._get_x_y(x, y)
        batch_size = x.size(1)
        y = y[..., :self.output_dim].reshape(self.horizon, batch_size,
                                             self.num_nodes * self.output_dim)
        return x, y

    # @profile
    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)

    def _compute_loss_single(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss_single(y_predicted, y_true)

    def _prepare_feature(self, x, y):
        x, y = self._get_feature(x, y)
        x, y = self._get_feature_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_feature(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        # CPU Version
        # x = torch.from_numpy(x).float()
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_feature_in_correct_dims(self, x, y):
        """batch_size
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x[..., :self.output_dim].reshape(self.seq_len, batch_size, self.num_nodes * self.output_dim)
        y = y[..., :self.output_dim].reshape(self.horizon, batch_size, self.num_nodes * self.output_dim)
        return x, y

    def calculate_zero_ratio(self, data):
        total_count = data.numel()  # 总元素数量
        zero_count = (data == 0).sum().item()  # 零值的数量
        zero_ratio = zero_count / total_count  # 计算零值比例
        return zero_ratio

    def check_stop_update(self, X, Y, threshold):
        zero_ratio_X = self.calculate_zero_ratio(X)
        zero_ratio_Y = self.calculate_zero_ratio(Y)
        if zero_ratio_X > threshold or zero_ratio_Y > threshold:
            # print("停止模型更新，因为X或Y中的零值比例超过阈值。")
            return True
        else:
            return False

    def find_matching_windows(self, Y_update, X_history, top_k):
        # Y_update shape: (12, 1, 207)
        # X_history_reduced shape: (12, 24000, 207)

        # 准备存储相似度
        X_history_reduced = X_history[:,:,:,0]
        similarities = torch.zeros(X_history_reduced.size(1), dtype=torch.float32)

        # 扩展 Y_update 的维度以匹配 X_history_reduced
        Y_update_expanded = Y_update.squeeze(1)  # shape: (12, 207)

        for idx in range(X_history_reduced.size(1)):  # 24000
            # Compute cosine similarity
            X_window = X_history_reduced[:, idx, :]
            dot_product = torch.sum(Y_update_expanded * X_window)
            norm_Y = torch.norm(Y_update_expanded)
            norm_X = torch.norm(X_window)

            similarities[idx] = (dot_product / (norm_Y * norm_X)).item()


        # 找到最小相似度的索引
        top_k_indices = torch.topk(similarities, top_k, largest=True).indices  # 取出前三个匹配
        # print(similarities[top_k_indices])
        return top_k_indices
        # if similarities[idx] > 0.80:
        #     return top_k_indices
        # else:
        #     return None

        # return top_k_indices


    def extract_matched_data(self, X_history, Y_history, top_k_indices):
        matched_X = []
        matched_Y = []

        for i in range(top_k_indices.size(0)):  # 对每个匹配的索引进行遍历
            indices = top_k_indices[i]
            matched_X.append(X_history[:, indices, :, :])  # shape: (12, 3, 207, 2)
            matched_Y.append(Y_history[:, indices])  # shape: (12, 3, 207)

        return torch.stack(matched_X, dim=1), torch.stack(matched_Y, dim=1)  # shape: (12, 3, 207, 2) 和 (12, 3, 207)

    def validation_olline(self, model0, model1, val_scale):
        with torch.no_grad():
            model0 = model0.eval()
            model1 = model1.eval()

            # 假设验证集有 10000 个样本，采样其中的 20%
            val_size = len(self._data['x_val'])  # 获取验证集的大小
            sample_size = int(val_scale * val_size)  # 抽取 20% 的样本

            # 随机抽取部分验证集数据的索引
            indices = torch.randperm(val_size)[:sample_size]

            # 使用 SubsetRandomSampler 来创建一个新的验证集 DataLoader
            val_sampler = SubsetRandomSampler(indices)
            val_loader = DataLoader(torch.utils.data.TensorDataset(self._data['x_val'], self._data['y_val']),
                                    batch_size=128, sampler=val_sampler)

            losses_0 = []
            losses_1 = []
            for x, y in val_loader:
                x, y = self._prepare_data(x, y)
                output0 = model0(x, self.sampledgraph)
                output1 = model1(x, self.sampledgraph)
                Loss_model0 = self._compute_loss(y, output0)
                Loss_model1 = self._compute_loss(y, output1)

                losses_0.append(Loss_model0.item())
                losses_1.append(Loss_model1.item())

            Loss_model0 = np.mean(losses_0)
            Loss_model1 = np.mean(losses_1)

        if Loss_model1 > Loss_model0:
            return True
        else:
            return False





