import psycopg2

class OutsourceMessageDao:
    def __openConnection(self):
        return psycopg2.connect(
                database="outsource_Message", 
                user="dev", 
                password="dev123", 
                host="148.251.15.86", 
                port="6432"
            )

    def selectMessagesByIds(self, ids):
      con = self.__openConnection()
      cur = con.cursor() 
      t = ', '.join(str(i[0]) for i in ids)
      cur.execute('''select t.message
                    from (select m.message,
                                row_number() over (partition by task_id order by id) as rn
                        from message as m
                        where m.task_id in ('''+t+''')
                            and m.direction = 'ToEmployee') as t
                    where t.rn = 1''')
        
      rows = cur.fetchall()

      con.close()  

      return rows