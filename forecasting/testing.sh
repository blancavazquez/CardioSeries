# Author: Blanca Vázquez
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Pipeline for time series forecasting (testing)"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"                          

python forecasting/testing.py --foldername "dilate_" \
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
      --subpopulation "nstemi" \
      --missing_values "True" \
      --saits_impute "True" \
      --model_name "CausalModel_epoch=48-val_loss=0.1126" \
      --feature_list "arterial_bp_mean respiratory_rate diastolic_bp spo2 heart_rate systolic_bp temperature"
