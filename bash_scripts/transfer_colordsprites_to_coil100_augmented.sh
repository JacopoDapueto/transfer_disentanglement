for ((i=0; i < 20; i++)); do 
	python3 dlib_execute_transfer_experiment.py --model_num="$i" --input_experiment WEAKCOLORDSPRITES --experiment COIL100AUGMENTED
	
done