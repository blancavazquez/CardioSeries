#Source github: https://github.com/CVxTz/time_series_forecasting
#source: https://towardsdatascience.com/how-to-use-transformer-networks-to-build-a-forecasting-model-297f9270e630

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Pipeline for time series forecasting"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

python forecasting/training.py --data_csv_path "data/" \
                               --output_json_path "metrics/trained_config" \
                               --log_dir "logs/" \
                               --model_dir "models/" \
                               --epochs "1000" \
                               --batch_size "32" \
                               --decay "1e-4" \
                               --database "iterative" \
                               --net "dcnn" \
                               --alpha "0.5" \
                               --gamma "0.01"  \
                               --window_size "24" \
                               --num_features "8" \
                               --conditions ""                       

#View logs: python3 -m tensorboard.main --logdir=models/
echo "End!!!"