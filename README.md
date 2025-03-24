# IonosphereModel
1.  module load stack/2024-06 python/3.11.6, see https://scicomp.ethz.ch/wiki/Python#Interactive_session
2.  python -m venv ./venv
3.  source venv/bin/activate
4.  pip install -r requirements.txt
TODO: Figure out
- cuda version that the gpus use. Get a torch version that is similar. We could ask arno to share with us a pip freeze > requirements.txt of his environment
- figure out how to tell sbatch 1-path to the code and 2-path to the python environment