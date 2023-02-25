import sys
from dotenv import load_dotenv


# Custom Classes
from classes.featureImportanceGenerator import FeatureImportanceGenerator
from classes.weightLogger import WeightLogger
from classes.modelsGenerator import ModelsGenerator


def main(params):
    if (params == 'importance'):
        FeatureImportanceGenerator()

    if (params == 'weights'):
        WeightLogger()

    if (params == 'generation'):
        ModelsGenerator()

if __name__ == '__main__':
    load_dotenv()
    if (len(sys.argv)) == 1:
        print('No argument provided')
    else:
        main(sys.argv[1])