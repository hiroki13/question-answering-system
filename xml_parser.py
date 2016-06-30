# -*- coding: utf-8 -*-

import sys
import gzip
from xml.etree import ElementTree
import cPickle
import random

from qa import QA, Question, Answer


def load(fn):
    print 'Load %s' % fn
    return ElementTree.parse(fn)


def save(fn, data):
    """
    :param fn: string; file name
    :param data: 1D: n_qa; elem: QA
    """
    print 'Save %s' % fn

    with gzip.open(fn + '.gz', 'wb') as gf:
        for qa in data:
            q_text = ''
            for q in qa.questions:
                q_text += 'Q\t%s\t%s\n' % (" ".join(q.title), " ".join(q.body))
            gf.writelines(q_text)

            a_text = ''
            for a in qa.answers:
                a_text += 'A\t%s\n' % " ".join(a.body)
            a_text += '\n'
            gf.writelines(a_text)


def save_binary(fn, data):
    """
    :param fn: string; file name
    :param data: 1D: n_qa; elem: QA
    """
    print 'Save %s' % fn

    with gzip.open(fn + '.gz', 'wb') as gf:
        for duplicates in data:
            q_text = ''
            for i, q in enumerate(duplicates):
                q_text += 'Q%d\t%s\t%s\n' % (i+1, " ".join(q.title), " ".join(q.body))
            q_text += '\n'
            gf.writelines(q_text)


def save_negative(fn, data):
    """
    :param fn: string; file name
    :param data: 1D: n_q; elem: xml_q
    """
    print 'Save %s' % fn

    q1_index = range(len(data))
    q2_index = range(len(data))
    random.shuffle(q1_index)
    random.shuffle(q2_index)

    with gzip.open(fn + '.gz', 'wb') as gf:
        for i in xrange(len(data)):
            q1 = data[q1_index[i]]
            q2 = data[q2_index[i]]

            q_text = 'Q1\t%s\t%s\n' % (" ".join(q1.title), " ".join(q1.body))
            q_text += 'Q2\t%s\t%s\n' % (" ".join(q2.title), " ".join(q2.body))
            q_text += '\n'
            gf.writelines(q_text)


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)


def separate_qa(posts):
    print '\nSEPARATE QA PAIRS'
    questions = []
    answers = []

    root = posts.getroot()

    max_post_id = -1
    for e in root:
        if e.attrib['PostTypeId'] == '1':
            questions.append(e)
            post_id = int(e.attrib['Id'])
            if max_post_id < post_id:
                max_post_id = post_id
        else:
            answers.append(e)

    print 'Questions: %d\tAnswers: %d' % (len(questions), len(answers))
    return questions, answers, max_post_id


def get_qa_pairs(questions, answers, max_post_id):
    print '\nGET QA PAIRS'

    print '\tCreating QA matrix...'
    qa_matrix = [None for i in xrange(max_post_id + 1)]
    for i, q in enumerate(questions):
        if i > 0 and i % 1000 == 0:
            print i,
            sys.stdout.flush()

        qa = QA()
        qa.questions.append(Question(q))
        qa_matrix[int(q.attrib['Id'])] = qa

    print '\n\tMatching questions and answers...'
    qa_matrix_len = len(qa_matrix)
    for i, a in enumerate(answers):
        if i > 0 and i % 1000 == 0:
            print i,
            sys.stdout.flush()

        try:
            parent_id = int(a.attrib['ParentId'])
            if qa_matrix_len > parent_id:
                qa = qa_matrix[parent_id]
                if qa is not None:
                    qa.answers.append(Answer(a))
        except:
            continue

    count = 0
    for qa in qa_matrix:
        if qa is not None and len(qa.answers) > 0:
            count += 1

    print '\nQA Pairs: %d' % count
    return qa_matrix


def get_link_set(links, check=False):
    print '\nCREATE LINK SETS'
    link_sets = []
    root = links.getroot()
    count_links = 0

    for e in root:
        post_id = e.attrib['PostId']
        r_post_id = e.attrib['RelatedPostId']
        link_type_id = e.attrib['LinkTypeId']

        if link_type_id != '3':
            continue

        count_links += 1

        for link_set in link_sets:
            if post_id in link_set or r_post_id in link_set:
                link_set.add(post_id)
                link_set.add(r_post_id)
                break
        else:
            l = [post_id, r_post_id]
            link_sets.append(set(l))

    if check:
        for k in link_sets:
            if len(k) > 1:
                print '%s' % str(k)

    print 'Binary Links: %d' % count_links
    print 'Link Sets: %d' % len(link_sets)
    return link_sets


def get_duplicate_links(links, check=False):
    print '\nCREATE BINARY LINKS'
    duplicate_links = []
    root = links.getroot()

    for e in root:
        post_id = e.attrib['PostId']
        r_post_id = e.attrib['RelatedPostId']
        link_type_id = e.attrib['LinkTypeId']

        if link_type_id == '3':
            duplicate_links.append((post_id, r_post_id))

    if check:
        for k in duplicate_links:
            if len(k) > 1:
                print '%s' % str(k)

    print 'Binary Links: %d' % len(duplicate_links)
    return duplicate_links


def merge_linked_qa(qa_pairs, link_sets):
    print '\nMERGE LINKED QA'
    merged_qa_pairs = []

    for link_set in link_sets:
        linked_qa = QA()

        for pair in qa_pairs:
            q = pair.questions[0]
            a = pair.answers

            if q.id in link_set:
                linked_qa.questions.append(q)
                linked_qa.answers.extend(a)
                pair.is_link = True

        if len(linked_qa.questions) > 1:
            merged_qa_pairs.append(linked_qa)

    for pair in qa_pairs:
        if pair.is_link is False:
            merged_qa_pairs.append(pair)

    print 'Merged QA Pairs: %d' % len(merged_qa_pairs)
    return merged_qa_pairs


def get_duplicate_qa_pairs(qa_matrix, duplicate_links):
    duplicate_pairs = []
    qa_matrix_len = len(qa_matrix)

    for link in duplicate_links:
        post_id = int(link[0])
        r_post_id = int(link[1])

        if post_id < qa_matrix_len and r_post_id < qa_matrix_len:
            qa1 = qa_matrix[post_id]
            qa2 = qa_matrix[r_post_id]
            if qa1 is not None and qa2 is not None:
                q1 = qa1.questions[0]
                q2 = qa2.questions[0]
                duplicate_pairs.append((q1, q2))

    print 'Duplicate Question Pairs: %d' % len(duplicate_pairs)
    return duplicate_pairs


def create_qa_retrieval_dataset(argv):
    print '\nQA RETRIEVAL DATA CREATION START\n'

    posts = load(argv.posts)
    links = load(argv.links)
    link_set = get_link_set(links, argv.check)

    q, a, max_post_id = separate_qa(posts)
    qa_pairs = get_qa_pairs(q, a, max_post_id)
    merged_qa_pairs = merge_linked_qa(qa_pairs, link_set)

    save(fn='test', data=merged_qa_pairs)


def create_seq_classification_dataset(argv):  # seq=semantically equivalent question
    print '\nSEMANTICALLY EQUIVALENT QUESTION CLASSIFICATION DATA CREATION START\n'

    posts = load(argv.posts)
    links = load(argv.links)
    duplicate_links = get_duplicate_links(links, argv.check)

    q, a, max_post_id = separate_qa(posts)
    qa_matrix = get_qa_pairs(q, a, max_post_id)
    duplicate_qa_pairs = get_duplicate_qa_pairs(qa_matrix, duplicate_links)

    save_binary(fn='test', data=duplicate_qa_pairs)


def create_seq_negative_samples(argv):
    print '\nSEQ CLASSIFICATION NEGATIVE DATA CREATION START\n'

    posts = load(argv.posts)
    q, a, max_post_id = separate_qa(posts)
    questions = [Question(u) for u in q]
    save_negative(fn='negative', data=questions)


def main(argv):
    if argv.task == 'binary':
        create_seq_classification_dataset(argv)
    elif argv.task == 'retrieval':
        create_qa_retrieval_dataset(argv)
    else:
        create_seq_negative_samples(argv)
