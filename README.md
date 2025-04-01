# IonosphereModel
In order to run execute:
1.  module load stack/2024-06 python/3.11.6
2.  python -m venv ./venv
3.  source venv/bin/activate
4.  pip install -r requirements.txt
5.  screen
6.  srun --ntasks=1 --cpus-per-task=10 --mem-per-cpu=4096 -G 1 -t 600 -o file.out -e file.err python main.py &


Then press Control+A+D

# Infrastructure / tools to look into:
- cuda version that the gpus use. Get a torch version that is similar. We could ask arno to share with us a pip freeze > requirements.txt of his environment
- figure out how to tell sbatch 1-path to the code and 2-path to the python environment
- Hector: Can we configure the code editor (e.g vscode) to run some python formatter (e.g black) on save? I find it makes code a lot nicer.
