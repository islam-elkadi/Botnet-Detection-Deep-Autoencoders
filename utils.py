from sys_utils import *

#------------------------------------------------------------------------------------------------#
#                                          Preprocessing                                         #
#------------------------------------------------------------------------------------------------#

def fisherScore(n_features,df):
    benign=df[df["anomaly"]==0].sample(n=1000,random_state=17)
    malicious=df[df["anomaly"]==1].sample(n=1000,random_state=17)
    temp_df=pd.concat([benign,malicious])
    score=fisher_score(temp_df.iloc[:,:-1].values,temp_df.iloc[:,-1].values)
    ranked_features=list(feature_ranking(score))
    ranked_features.append(n_features)
    return ranked_features

def data_partitions(df,n):

    # Fisher score
    idx_order=fisherScore(n,df)

    # Create & scale data partitions
    benign_df=df[df["anomaly"]==0].iloc[:,idx_order]
    train,validate,test=np.split(benign_df.sample(frac=1,random_state=42),[int(1/3*len(benign_df)),int(2/3*len(benign_df))])
    
    train.iloc[:,:-1]=scaler.fit_transform(train.iloc[:,:-1].values)
    validate.iloc[:,:-1]=scaler.fit_transform(validate.iloc[:,:-1].values)

    test_set=pd.concat([test,df[df["anomaly"]==1].iloc[:,idx_order]],sort=True,ignore_index=True)
    test_set.iloc[:,:-1]=scaler.transform(test_set.iloc[:,:-1].values)

    return train,validate,test_set

#------------------------------------------------------------------------------------------------#
#                                   Modeling: Deep Autoencoders                                  #
#------------------------------------------------------------------------------------------------#

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

def train_autoencoder(input_dim,complie_kwags,fit_kwags,path):
    model=auto_encoder(input_dim)
    model.compile(complie_kwags)
    model.fit(fit_kwags)
    model.save(path)
    return model

def thresh_autoencoder(model,validate):
    # Anomaly threshold
    validate_pred=model.predict(validate)
    mse=np.mean(np.power(validate-validate_pred,2),axis=1)
    threshold=mse.mean()+mse.std()
    return threshold

def test_autoencoder(model,threshold):
    # Predict anomalies
    test_pred=model.predict(test.iloc[:,:-1]) 
    mse=np.mean(np.power(test.iloc[:,:-1]-test_pred,2),axis=1)
    predictions=(mse>threshold).astype(int)

    # Performance metrics
    metrics={
        "accuracy":accuracy_score(test.iloc[:,-1],predictions),
        "recall":recall_score(test.iloc[:,-1],predictions),
        "precision":precision_score(test.iloc[:,-1],predictions)
    }

    return metrics
