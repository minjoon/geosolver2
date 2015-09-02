import sys
from geosolver.database.geoserver_interface import geoserver_interface
from geosolver.database.utils import zip_diagrams

__author__ = 'minjoon'

def test_geoserver_interface():
    data = geoserver_interface.download_questions(["annotated"])
    ann = geoserver_interface.download_semantics()
    print(ann)
    print(data)


def test_zip_diagrams():
    questions = geoserver_interface.download_questions(['development'])
    zip_diagrams(questions, '/Users/minjoon/Desktop/development.zip')



if __name__ == "__main__":
    test_geoserver_interface()
    # test_zip_diagrams()