for ((i=0; i < 20; i=i+1)); do
	python3 dlib_execute_weakly_supervised_experiment.py --model_num=$i --experiment WEAKCOIL100AUGMENTED 
	
done

