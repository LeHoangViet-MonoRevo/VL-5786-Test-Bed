import pymysql
from elasticsearch import Elasticsearch

from config import settings


def mysql_physical_obj_connection():
    conn = pymysql.connect(
        host=settings.PHYSICAL_OBJ_DATABASE_HOST,
        user=settings.PHYSICAL_OBJ_DATABASE_USER,
        password=settings.PHYSICAL_OBJ_DATABASE_PASSWORD,
        database=settings.PHYSICAL_OBJ_DATABASE,
    )
    return conn


def es_connection():
    es = Elasticsearch(
        settings.ELASTICSEARCH_ADDRESS,
        basic_auth=(settings.ELASTICSEARCH_USERNAME, settings.ELASTICSEARCH_PASSWORD),
    )

    return es.options(request_timeout=60)


es_client = es_connection()


mysql_physical_obj_connect = mysql_physical_obj_connection()
