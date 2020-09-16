from dao import OutsourceTaskDao
from dao import OutsourceMessageDao
import pymorphy2
import re
import html2text
import os

class TaskDataSetService:
    taskDao = OutsourceTaskDao()
    messageDao = OutsourceMessageDao()

    def generateFiles(self):
        categories = ['Payroll', 'CallbackClient', 'DocumentRequest', 'LoadingUnloading1c', 'QuestionToAccountant']

        for category in categories:
            ids = self.taskDao.selectTaskIdsByCategory(category, 10000)
            messages = self.messageDao.selectMessagesByIds(ids)
            lines = [self.__clean_text(message[0]) for message in messages]

            self.__writeFile(category, lines)
            print('writed - ', category)

        print('finish')

    def __clean_text(self, text):
        ma = pymorphy2.MorphAnalyzer()

        text = html2text.html2text(text)
        text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
        text = text.lower()
        text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
        text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols  
        # text = " ".join(ma.parse(str(word))[0].normal_form for word in text.split())
        # text = " ".join(str(word) for word in text.split())
        text = ' '.join(word for word in text.split() if len(word)>2)
        # text = text.encode("utf-8")

        return text
    
    def __writeFile(self, fileName, lines):
        filePath = '/mnt/c/rrr/test/'+fileName+'.txt' 

        os.remove(filePath)
        f = open(filePath, 'w+')
        
        for line in lines:
            f.write(line + '\n')