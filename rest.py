from utils import *

app=Flask(__name__)

#------------------------------------------------------------------------------------------------#
#                                       Error handling and test                                  #
#------------------------------------------------------------------------------------------------#

@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({"error": "Bad request"}),400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Not found"}),404)

@app.route("/",methods=["GET"])
def test_app():
    return jsonify({"success": "true"})

#------------------------------------------------------------------------------------------------#
#                                            Modeling                                            #
#------------------------------------------------------------------------------------------------#

@app.route("/data_partitions",methods=["POST"])
def data_partitions():
    # Load data
    data=request.files["data"]
    df=pd.read_csv(data)
    df=df.loc[:,~df.columns.str.contains("^Unnamed")]

    # Fisher score
    idx_order=fisherScore(115,df)

    # Create & scale data partitions
    benign_df=df[df["anomaly"]==0].iloc[:,idx_order]
    train,validate,test=np.split(benign_df.sample(frac=1,random_state=42),[int(1/3*len(benign_df)),int(2/3*len(benign_df))])
    
    train.iloc[:,:-1]=scaler.fit_transform(train.iloc[:,:-1].values)
    validate.iloc[:,:-1]=scaler.fit_transform(validate.iloc[:,:-1].values)

    test_set=pd.concat([test,df[df["anomaly"]==1].iloc[:,idx_order]],sort=True,ignore_index=True)
    test_set.iloc[:,:-1]=scaler.transform(test_set.iloc[:,:-1].values)

    # Save partitioned data
    train.to_csv("training_scaled.csv")
    validate.to_csv("validation_scaled.csv")
    test_set.to_csv("testing_set_scaled.csv")

    return jsonify({"Partitions created":"Success"})

@app.route("/train_deep_autoencoder",methods=["POST"])
def train_deep_autoencoder():

    # # Request body
    # train=request.files["train"]
    # train=pd.read_csv(train)
    # validate=request.files["validate"]
    # validate=pd.read_csv(validate)

    # Auto-encoder
    tensor_board=TensorBoard(log_dir="./logs/",histogram_freq=0,write_graph=True,write_images=True)
    model=auto_encoder(115)
    model.compile(loss="mean_squared_error",optimizer="sgd",metrics=["mse","mae","mape"])
    model.fit(train,train,epochs=64,batch_size=100,verbose=1,validation_data=(validate,validate),callbacks=[tensor_board])

    model.save("./models/autoencoder.h5")

    return jsonify({"Training":success})

@app.route("/test_deep_autoencoder",methods=["GET"])
def test_deep_autoencoder():
    # Request body
    test=request.files["test"]
    validate=request.files["validate"]

    # # Load data
    # test=pd.read_csv("testing_set_scaled.csv")
    # validate=pd.read_csv("validation_scaled.csv")

    # Load model
    model=load_model("./models/autoencoder.h5")

    # Anomaly threshold
    validate_pred=model.predict(validate)
    mse=np.mean(np.power(validate-validate_pred,2),axis=1)
    threshold=mse.mean()+mse.std()

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

    return jsonify({"Performance":metrics})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)