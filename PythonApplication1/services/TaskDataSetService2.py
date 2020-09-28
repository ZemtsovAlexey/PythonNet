from dao import OutsourceTaskDao
from dao import OutsourceMessageDao
import pymorphy2
import re
import html2text
import os
import glob
import csv

class TaskDataSetService2:
    taskDao = OutsourceTaskDao()
    messageDao = OutsourceMessageDao()

    def generateFiles(self):
        categories = ['Payroll', 'CallbackClient', 'DocumentRequest', 'LoadingUnloading1c', 'QuestionToAccountant']
        # categories = ['CallbackClient']

        filePath = r'C:\rrr\test\\data.csv'

        if os.path.exists(filePath):
            os.remove(filePath)

        with open(filePath, 'w+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Category", "Description"])

            for category in categories:
                ids = self.taskDao.selectTaskIdsByCategory(category, 50000)
                messages = self.messageDao.selectMessagesByIds(ids, 1000)
                lines = [self.__clean_text(message[0]) for message in messages]
                
                for line in lines:
                    if (len(line) > 0):
                        writer.writerow([category, line])

                print('writed - ', category)

        print('finish')

    def __clean_text(self, text):

        text = html2text.html2text(text)
        text = text.replace("\\", " ")
        text = text.lower()
        text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', ' ', text)
        text = re.sub('\W|[_]', ' ', text)
        text = ' '.join(word for word in text.split() if len(word)>2)

        return text

    def __writeFileTxt(self, category, number, text):
        filePath = r'C:\rrr\test\\train\\'+category+'\\' + category + '_' + str(number) + '.txt' 

        if os.path.exists(filePath):
            os.remove(filePath)

        f = open(filePath, 'w+', encoding='utf-8')
        f.write(text)
    
    def __writeFile(self, fileName, lines):
        filePath = r'C:\rrr\test\\'+fileName+'.csv' 

        if os.path.exists(filePath):
            os.remove(filePath)

        with open(filePath, 'w+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Category", "Description"])
        
            for line in lines:
                if (len(line) > 0):
                    writer.writerow([fileName, line])