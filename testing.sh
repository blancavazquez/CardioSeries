#Source github: https://github.com/CVxTz/time_series_forecasting
#source: https://towardsdatascience.com/how-to-use-transformer-networks-to-build-a-forecasting-model-297f9270e630

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Pipeline for time series forecasting"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"                          

python forecasting/testing.py --foldername "prueba" \
      --batch_size "32" \
      --net "causal_dcnn" \
      --database "mimic" \
      --alpha "0.3" \
      --gamma "0.01"  \
      --input_seq_length "6" \
      --output_seq_length "6" \
      --output_rnn "100" \
      --num_features "7" \
      --conditions "5" \
      --subpopulation "stemi" \
      --missing_values "True" \
      --saits_impute "True" \
      --model_name "CausalModel_epoch=00-val_loss=0.1790" \
      --feature_list "arterial_bp_mean respiratory_rate diastolic_bp spo2 heart_rate systolic_bp temperature"