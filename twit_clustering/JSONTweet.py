#!/usr/bin/env python3
__author__ = 'Kristy'

'''Script to read the original JSON file format of twitter JSON and extract info.
You can choose what fields to print out and they are tab-separated.
You can also choose to mask usernames, urls and hashtags for anonymity if needed.
'''

import json
import re

class JSONTweet:
    def __init__(self, json_text):
        """Takes a subset of the twitter JSON format and saves these as attributes, cleaning whitespace."""
        td = json.loads(json_text) #LOAD THE TWEET
        self.id = td['id']
        self.text = re.sub('[\n\t]', ' ', td['text'])  #remove newlines in the tweet
        self.created_at = td['created_at']
        self.timestamp = td['timestamp']
        self.in_reply_to_status_id = td['in_reply_to_status_id']
        self.retweet_count = td['retweet_count']

    def __str__(self):
        return("id: {}, text: {}, reply: {}, retweeted: {}"
              .format(str(self.id), self.text, self.in_reply_to_status_id, self.retweet_count))

def json_to_string(linestring, mask = (0, 0, 0), fields = None):
    """Take a line from the twitter API, return what should be printed as output."""
    #print("this tweet is", linestring)
    # if fields is None: #default is set outside the function
    #     fields = ['id', 'text']
    current_tweet = JSONTweet(linestring)
    if mask[0]: #hashtag
        current_tweet.text =  re.sub(r"(?<!\w)\#[^\s]+",'#hash', current_tweet.text)
    if mask[1]: #url
        current_tweet.text = re.sub(r"http:\S+", '/url/', current_tweet.text)
    if mask[2]: #username
        current_tweet.text = re.sub(r"(?<!\w)\@[^\s]+", '@usr', current_tweet.text)
    outputstring = '\t'.join([str(getattr(current_tweet, x)) for x in fields])+'\n'
    del current_tweet
    return outputstring

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Read tweets from a json file and save in tab separator utf-8 with field info.")
    #input file, output directory
    parser.add_argument('input_json_file', help='input twitter API json file', type=str)
    parser.add_argument('-output_prefix', '-o', help='output location prefix, if current directory then just name',
                        default='split_docs')
    #how to split the lines
    parser.add_argument('--percent_train', '-rtra', help='Ratio training', type=int, default=1)
    parser.add_argument('--percent_dev', '-rdev', help='Ratio to dev file', type=int, default=0)
    parser.add_argument('--percent_test', '-rtes', help='Ratio to test file', type=int, default=0)
    parser.add_argument('--block_split', '-bs', help='If True splits the corpus retaining order and in large blocks, '
                                              'rather than still retaining order but taking dev/test throughout whole doc.',
                        action='store_true')
    #anonymising options
    parser.add_argument('-mask_hash', help="Anonymise hashtags", action='store_true')
    parser.add_argument('-mask_url', help="Anonymise url", action='store_true')
    parser.add_argument('-mask_user', help="Anonymise user", action='store_true')
    #fields to print in output doc
    parser.add_argument('-fields', help='List json fields to write, from text, id, created_at, timestamp, in_reply_to_status_id, retweet_count',
                        nargs='+', type=str, default=[])
    parser.add_argument('-headeroff', help='Turn of the header line explaining the fields', action='store_true')

    args = parser.parse_args()
    print(args)

    #handle where lines split
    split_ratio = [args.percent_train, args.percent_dev, args.percent_test]
    print("Splitting the file {}; saving in dir {}\n with {} test :  {} dev : {}test lines".
          format(args.input_json_file, args.output_prefix, split_ratio[0], split_ratio[1], split_ratio[2]))

    if args.block_split: #this is annoying and requires extra reading
        totaldocs = sum(1 for line in open(args.input_json_file))
        print("Your input has {} lines".format(totaldocs))
        block_lines = [(x * totaldocs)//sum(split_ratio) for x in split_ratio]
        block_lines[-1] += totaldocs - sum(block_lines)
    else:
        block_lines = split_ratio


    #determine masking
    maskingoptions = [args.mask_hash, args.mask_url, args.mask_user]

    #determine fields printed
    possiblefields = ['text', 'id', 'created_at', 'timestamp', 'in_reply_to_status_id', 'retweet_count']
    printingfields = [x for x in args.fields if x in possiblefields]
    if len(printingfields)==0:
        printingfields = ["id", "text"] #default fields and order

    #read the input file and write output
    with open(args.input_json_file, 'r', encoding='utf-8') as finput:
        ftrain, fdev, ftest = [open(args.output_prefix+'.'+ x[1], 'w', encoding='utf-8')
                               for x in zip(split_ratio, ['train', 'dev','test'])]# if x[0] > 0]
        file_holder = [ftrain, fdev, ftest]
        if not args.headeroff:
            for f in file_holder:
                f.write(str('\t'.join(printingfields)+'\n'))

        #line_counter = 0;
        eof=False
        while eof==False:
            for j in range(3): #for train, dev, test
                for i in range(block_lines[j]): #until consec lines exceeded
                    #line_counter +=1; print(line_counter)
                    line_content = finput.readline()
                    if not line_content:
                        eof=True
                        break
                    else:
                        file_holder[j].write(json_to_string(line_content, maskingoptions , printingfields))
    for f in file_holder:
        f.close()

#example command:
'''./JSONTweet original_json_file ./output/july -rtra 8 -rdev 1 -rtest 1 -mask_url -mask_user -fields id text '''
'''-bs'''#if taking large temporally adjacent blocks

if __name__=="__main__":

    main()