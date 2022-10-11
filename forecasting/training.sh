# Author: Blanca Vázquez
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Pipeline for time series forecasting (training)"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

python forecasting/training.py --foldername "dilate_" \
       --epochs "1000" \
       --batch_size "32" \
       --learning_rate "0.01" \
       --decay "1e-8" \
       --net "CausalModel" \
       --alpha "0.3" \
       --gamma "0.01"  \
       --kfold "5"  \
       --input_seq_length "6" \
       --output_seq_length "6" \
       --output_rnn "100" \
       --num_features "7" \
       --conditions "5" \
       --missing_values "True" \
       --saits_impute "True" \
       --feature_list "arterial_bp_mean respiratory_rate diastolic_bp spo2 heart_rate systolic_bp temperature"
