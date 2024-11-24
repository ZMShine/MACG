import argparse
import numpy as np
import os
import pandas as pd

def split_data(args):
    # for .h5 datasets
    df = pd.read_hdf(args.data_filename)

    # for .csv datasets
    # df = pd.read_csv(args.data_filename)

    # for .npz datasets
    # df = np.load(args.data_filename)['data']
    # df = pd.DataFrame(df[:,:,0], index=None)

    #初始化x和y的序列长度
    x_offsets = np.sort(#排序
        np.concatenate((np.arange(-11,1,1),))#拼接
    )
    y_offsets = np.sort(np.arange(1,13,1))
    #print(x_offsets, y_offsets)
    #[-11 -10  -9  -8  -7  -6  -5  -4  -3  -2  -1   0]to [ 1  2  3  4  5  6  7  8  9 10 11 12]

    #根据映射关系构建序列对应的数据
    x,y = gen_seq2seq(
        df,
        x_offsets = x_offsets,
        y_offsets = y_offsets,
        add_timestep =True,
    )

    print("X_shape:", x.shape, "Y_shape:", y.shape)

    #split data
    num_samples = x.shape[0]
    num_train = round(num_samples * 0.7)
    num_test = round(num_samples * 0.2)
    num_val = num_samples - num_train - num_test
    # num_val = round(num_samples * 0.1)


    #training set:
    x_train, y_train = x[:num_train], y[:num_train]

    #validation set:
    x_val, y_val = (
        x[num_train: num_train+num_val],
        y[num_train: num_train + num_val],
    )
    #testiong set:
    x_test, y_test = x[-num_test:], y[-num_test:]

    #store data in .NPZ file for training, validation and testing
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
        )

    print("The data splitting has been completed.")


def gen_seq2seq(df, x_offsets, y_offsets, add_timestep = True):
    """
        Generate samples from
        :param df:
        :param x_offsets:
        :param y_offsets:
        :param add_timestep:
        :return:
        # x: (epoch_size, input_length, num_nodes, input_dim)
        # y: (epoch_size, output_length, num_nodes, output_dim)
        """
    print(df.shape)
    # num_samples, num_nodes = df.shape#确定样本量，序列数量
    num_samples = df.shape[0]
    num_nodes = df.shape[1]
    #数据末尾增加一维，data.shape = (num_samples, num_nodes, 1)
    #df = df.fillna(df.interpolate())# 上下均值填充空值
    data = np.expand_dims(df.values, axis = -1)#axis=0在第一维操作，axis=1在第二维操作，axis=-1在最后一维操作，axis代表的是多维数组中数据操作的方向
    data_list = [data]

    #如果序列缺少时间索引，手动增加
    if df.index.dtype == "int64":
        df["time"]=pd.date_range('03/02/2015T00:00:00',periods=len(df),freq='1H')
        df.index = df["time"]

    if add_timestep:
        #时间戳标准化，整个时间轴缩放到0到1内
        #df.index.values：'2017-01-01T00:00:00.000000000'
        #df.index.values.astype("datetime64[D]")：‘2017-01-01’
        #np.timedelta64(1,"D")：1 days
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]"))/np.timedelta64(1,"D")
        #按序列数量复制时间戳
        timestep = np.tile(time_ind,[1, num_nodes, 1]).transpose((2, 1, 0))#np.tile(;复制成几维&某一维复制多少)
        #将标准化的时间戳加到数据列表末尾
        data_list.append(timestep)
    #拼接原始数据和时间戳
    data = np.concatenate(data_list, axis=-1)
    # print(data.shape)
    #current data shape [样本量， 序列个数， 2] 2 refers to 样本值和时间戳
    x,y =[], []

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples-abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, :, :]
        y_t = data[t + y_offsets, :, :]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    #current x shape [num_samples, x_len, stream_num, sample_value+time(2)]
    return x, y
if __name__ == "__main__":
    #定义相关默认参数
    parser = argparse.ArgumentParser()#用于命令行传参
    parser.add_argument(
        "--output_dir", type=str, default="../data/Processed Data", help="output directory."
    )#type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument(
        "--data_filename", type=str, default="../data/pems-bay.h5", help="Dataset Reading.",
    )
    args = parser.parse_args()#获得传入的参数

    #执行数据分割
    print("Dataset is been splitting...")
    # print(args.data_filename)
    split_data(args)
