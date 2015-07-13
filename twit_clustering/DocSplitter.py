__author__ = 'kristy'

'''Contains classes DocSplitter(an input document) and TweetText'''
import re

class DocSplitter(object):
    '''Takes the file with one document per line, yields the fields, and splits text into list form.
     Some options to mask numbers, case conversions etc.'''

    def __init__(self, filename, splitpunct=True, masknumbers=True, preserveupper=False):
        #file being opened is already a document object
        if isinstance(filename, self.__class__):
            self.__init__(filename.nickname)
        else:
        #the file is fresh and unopened
            self.file = open(filename, 'r', encoding='utf-8') #file object
            self.nickname = str(filename) #filename as a string

            self.fieldnames = self.file.readline().strip('\n').split('\t')
            print('Fieldnames are :', self.fieldnames)
            self.no_fields = len(self.fieldnames)
            self.split_punct = splitpunct
            self.mask_nums = masknumbers
            self.preserve_upper=preserveupper


        #attmpt to overwrite values if found
            try:
                self.text_location = self.fieldnames.index('text')
            except:
                self.text_location = 1
            try:
                self.id_location = self.fieldnames.index("id")
            except:
                self.id_location=0 #defaults
            print("Text occurs in position {}".format(self.text_location))

            self.eof = False
            self.linesread = 0
            print('Document presumed to have headers, moving to position 1')
            self.file.seek(0,0)
            self.file.readline()

    def checkeof(self):
        if self.eof:
            print("This operation cannot be performed, the corpus has been read")
            return None
        else:
            return False

    def __str__(self):
        return "Document object from the file {}".format(self.nickname)

    def seekFileStart(self):
        '''self.file becomes file object at line 0'''
        if self.file.closed:
            print('reopening corpus file')
            self.file = open(self.nickname,'r', encoding='utf-8')
            self.file.readline()
            self.eof=False
        else:
            print("Document presumed to have headers")
            self.file.seek(0,0)
            self.file.readline()

    def giveLineWhole(self):
        '''Return the next line in the doucument, as string'''
        line = self.file.readline()
        if line.startswith("{"):
            print("WARNING:\nThis file is still in JSON format. Use a script to split it into utf-8 tab-separated."
                  "\nThe script will now exit.")
            exit()
        elif line =='' or self.eof:
            print("End of sentence file reached")
            self.file.close()
            self.eof = True
            return None
        else:
            self.linesread +=1
            return line

    def giveTextSplit(self, giveId=False):
        '''Return the next line as a list, or as list of [id, text-in-list]'''

        rawline = self.giveLineWhole()
        if self.eof ==False:
            rawline = rawline.split('\t')
            if len(rawline) < self.no_fields:
                print("The input stream has too many few fields. \nPut a correct header on the file or clean your input.")
                exit()
            id = rawline[self.id_location]
            textstring = rawline[self.text_location]
            #print(textstring)
            text = textToList(textstring, splitpunct=self.split_punct, masknums=self.mask_nums, presup=self.preserve_upper)
            if giveId:
                return id, text
            else:
                return text

    def giveWholeCorpus(self, giveId=False):
        '''Return an array with the whole corpus, add giveId if this should be returned as a separate vector'''
        self.seekFileStart()
        print("Warning, the whole corpus is being split to a list. This is slow.")
        corpus = []; ids = []
        newline = self.giveTextSplit(giveId=True)
        while self.eof == False and newline is not None:
            newid, newtext = newline[0], newline[-1] #if no id field, then this is a copy of text
            corpus.append(newtext)
            ids.append(newid)
            newline = self.giveTextSplit(giveId=True)
        if giveId:
            return ids, corpus
        else:
            return corpus

    def giveRawTweetObj(self):
        '''Give a Tweet Object with info about the line of text.
        Useful when more fields than just id and text are included.'''
        fields = self.fieldnames
        data = self.giveLineWhole().split('\t')
        return TweetObject(zip(fields, data))

class TweetObject(object):
    '''Something saving info about a tweet, might later include calculated values/visualisation info.
    Functions best where the input document has headers, one called 'text'. '''
    def __init__(self, args):
        for x in args:
            #takes args as zipped data such as zip(fieldnames, data)
            setattr(self, x[0], x[1])
        if getattr(self, 'text'):
            self.split_text = textToList(self.text)
        elif getattr(self, 'TEXT'):
            self.split_text = textToList(self.TEXT)
        elif getattr(self, 'Text'):
            self.split_text = textToList(self.Text)
    def __repr__(self):
        return str([x for x in self.__dir__() if x.startswith('__')==False])


def textToList(text, splitpunct=True, masknums=True, presup=False):
    if not type(text)==str:
        print("This can only split a string")
        TypeError
        return None
    if splitpunct:
        text = re.sub(r'(?<=\w)([^\d\s\w])(?=\s)', r' \1', text)
    if masknums:
        text = re.sub(r'[0-9]', r'5', text)
    if not presup:
        text = text.lower()
    text = re.split(r'\s+', text) #splits at all whitespace
    return [t for t in text if t !='']  #TODO: This iteration removes the empty string at the end of the list. Why is this occurring?

if __name__=='__main__':
    mydoc = DocSplitter('split_docs.train')
    aline = mydoc.giveLineWhole()
    print('a', aline)
    bline = mydoc.giveTextSplit()
    print('b',bline)
    cline = mydoc.giveRawTweetObj()
    print('c',cline)
    dline = mydoc.giveWholeCorpus()
    print('d', dline)
    eline = mydoc.giveWholeCorpus(giveId=True)
    print('e', eline)