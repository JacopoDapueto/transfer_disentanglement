for ((i=0; i < 20; i++)); do 
	python3 dlib_execute_transfer_experiment.py --model_num="$i" --input_experiment WHITECOLORDSPRITES --experiment COIL100AUGMENTEDBINARY
	
done
