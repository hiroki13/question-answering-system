from tokenizer import tokenize


class QA(object):
    def __init__(self):
        self.is_link = False
        self.questions = []
        self.answers = []


class Question(object):
    def __init__(self, q):
        self.id = q.attrib['Id']
        self.title = tokenize(q.attrib['Title'])
        self.body = tokenize(q.attrib['Body'])


class Answer(object):
    def __init__(self, a):
        self.parent_id = a.attrib['Id']
        self.body = tokenize(a.attrib['Body'])



