from sys_utils import *

def fisherScore(n_features,df):
    benign=df[df["anomaly"]==0].sample(n=1000,random_state=17)
    malicious=df[df["anomaly"]==1].sample(n=1000,random_state=17)
    temp_df=pd.concat([benign,malicious])
    score=fisher_score(temp_df.iloc[:,:-1].values,temp_df.iloc[:,-1].values)
    ranked_features=list(feature_ranking(score))
    ranked_features.append(n_features)
    return ranked_features

def auto_encoder(input_dim):
    autoencoder=Sequential()
    autoencoder.add(Dense(int(0.75*input_dim),activation="relu",input_shape=(input_dim,)))
    autoencoder.add(Dense(int(0.5*input_dim),activation="relu"))
    autoencoder.add(Dense(int(0.33*input_dim),activation="relu"))
    autoencoder.add(Dense(int(0.25*input_dim),activation="relu"))
    autoencoder.add(Dense(int(0.33*input_dim),activation="relu"))
    autoencoder.add(Dense(int(0.5*input_dim),activation="relu"))
    autoencoder.add(Dense(int(0.75*input_dim),activation="relu"))
    autoencoder.add(Dense(input_dim))
    return autoencoder

if __name__ == "__main__":
    df=pd.read_csv("dataset.csv")
    print(fisherScore(df.shape[1]-1,df))