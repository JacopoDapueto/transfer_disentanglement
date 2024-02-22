import pickle




if __name__=="__main__":

    with open('info.pkl', 'rb') as f:
        train_config = pickle.load(f)

        print(train_config)
