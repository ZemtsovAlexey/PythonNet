import psycopg2

class OutsourceTaskDao:
    """description of class"""

    def __openConnection(self):
        return psycopg2.connect(
                database="outsource_Task", 
                # user="postgres", 
                # password="123456", 
                # host="127.0.0.1", 
                # port="5432"
                user="dev", 
                password="dev123", 
                host="148.251.15.86", 
                port="6432"
            )

    def selectTaskIds(self):
      con = self.__openConnection()
      cur = con.cursor() 
      cur.execute("SELECT id, title from task limit 10")
        
      rows = cur.fetchall()

      con.close()  

      return rows

    def selectTaskIdsByCategory(self, category, limit):
      con = self.__openConnection()
      cur = con.cursor() 
      cur.execute("select id from task where category = %(category)s limit %(limit)s", {'category': category, 'limit': limit})
      rows = cur.fetchall()
      con.close()  

      return rows

    def typesCount(self):
      con = self.__openConnection()
      cur = con.cursor() 
      cur.execute("select category, count(id) as \"count\" from task group by category")
        
      rows = cur.fetchall()

      con.close()  

      return rows

# taskDao = OutsourceTaskDao()
# rows = taskDao.typesCount()

# print(rows)
# for row in rows:  
#   print("Id =", row[0], "; Title =", row[1])