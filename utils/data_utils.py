from scipy.io import loadmat

def get_data_tag(faultP=4, desample=512):
    filename = [f".\\data\\processed\\fault_probability_{i}.mat" for i in [15, 25, 50, 75, 100]]
    dataset = loadmat(filename[faultP])
    data = dataset["dataset"]
    tag = dataset["datatag"].flatten()
    step = int(data.shape[0] / desample)
    data = data[::step,:,:]
    # data = np.transpose(data, (2, 0, 1))
    return data, tag


if __name__ == '__main__':
    data, tag = get_data_tag()
    print(data.shape, tag.shape)