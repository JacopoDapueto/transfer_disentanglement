for ((i=0; i < 20; i=i+1)); do
	python3 dlib_train_source.py --model_num=$i --experiment WEAKCOLORDSPRITES --grad_acc_steps 1
	
done
