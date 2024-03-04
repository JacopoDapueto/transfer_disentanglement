for ((i=0; i < 20; i=i+1)); do
	python3 dlib_execute_transfer_experiment.py --model_num="$i" --input_experiment WEAKCOIL100AUGMENTEDBINARY --experiment RGBDOBJECTS
	
done
