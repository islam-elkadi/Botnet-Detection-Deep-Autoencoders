from utils import *

class deep_autoencoders():

    def __init__(n,fit_kwags,complie_kwags):
        self._n=n
        self._fit_kwags=json.loads(fit_kwags)
        self._complie_kwags=json.loads(complie_kwags)

    def autoencoder_main(train,validate,test,path): 
        # Train model or load-pretrained model
        if train:
            model=train_autoencoder(
                self._n,
                self._complie_kwags,
                self._fit_kwags,
                train,
                validate,
                path
            )
        else:
            model=load_model(path)

        threshold=thresh_autoencoder(model,validate)
        metrics=test_autoencoder(test,model,threshold)
        return metrics     

if __name__=="__main__":

    # Input Parameters
    parser=argparse.ArgumentParser(description="Arguments for deep autoencoder class")
    parser.add_argument("-d","--data",required=True,type="str",help="Path to dataset")
    parser.add_argument("-m","--model",required=True,type="str",help="Select model to work with")
    parser.add_argument("-t","--train",required=True,type="str",help="Train or load pre-trained model")
    parser.add_argument("-f","--fit",required=False,type="dict",help="Autoencoder training Parameters")
    parser.add_argument("-c","--compile",required=True,type="dict",help="Autoencoder compiling parameters")
    parser.add_argument("-mp","--model_path",required=True,type="str",help="Load model name or save as model name")

    # Load data
    df=pd.read_csv(args.path)
    train,validate,test_set=data_partitions(df,len(df)-1)

    if args.model == "autoencoder":
        autoencoders=deep_autoencoders(len(df)-1,args.fit_kwags,args.complie_kwags)
        metrics=autoencoders.autoencoder_main(train,validate,test,args.model_name)
        print(metrics)