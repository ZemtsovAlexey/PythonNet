import mysql.connector
from contextlib import closing
#from pymysql.cursors import DictCursor

class CrmDocuments:
    """description of class"""
    
    def selectDocumentIds(self):
        connection = mysql.connector.connect(
            host='195.201.242.96',
            user='crm_read',
            passwd='U2q78dRsOnxrNqzJnTNYUmHC',
            database='crm_prod',
        )

        cursor = connection.cursor(buffered=True)
        query = ("SELECT id from documents where document_name  = 'image.png' and created_by <> '7aad6953-41c4-e07c-1415-589c59984165';")
        cursor.execute(query)

        result = [str(row[0]) for row in cursor]

        connection.close()

        return result

        #with closing(mysql.connector.connect(host='195.201.242.96',
        #    user='crm_read',
        #    passwd='U2q78dRsOnxrNqzJnTNYUmHC',
        #    database='crm_prod')) as connection:
        #    with connection.cursor() as cursor:
        #        query = """
        #        SELECT *
        #        from documents
        #        where document_name  = 'image.png' and created_by <> '7aad6953-41c4-e07c-1415-589c59984165'
        #        limit 1000;
        #        """
        #        cursor.execute(query)
        #        for row in cursor:
        #            print(row)

