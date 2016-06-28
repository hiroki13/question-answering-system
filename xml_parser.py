# -*- coding: utf-8 -*-

import gzip
from xml.etree import ElementTree

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


def separate_qa(posts):
    print '\nSEPARATE QA PAIRS'
    questions = []
    answers = []

    root = posts.getroot()

    for e in root:
        if e.attrib['PostTypeId'] == '1':
            questions.append(e)
        else:
            answers.append(e)

    print 'Questions: %d\tAnswers: %d' % (len(questions), len(answers))
    return questions, answers


def get_qa_pairs(questions, answers):
    print '\nGET QA PAIRS'
    qa_pairs = []

    for q in questions:
        qa = QA()
        qa.questions.append(Question(q))
        qid = qa.questions[0].id

        for a in answers:
            if a.attrib['ParentId'] == qid:
                qa.answers.append(Answer(a))

        if len(qa.answers) > 0:
            qa_pairs.append(qa)

    print 'QA Pairs: %d' % len(qa_pairs)
    return qa_pairs


def get_link_set(links, check=False):
    print '\nCREATE LINK SETS'
    link_set = []
    root = links.getroot()

    for e in root:
        post_id = e.attrib['PostId']
        r_post_id = e.attrib['RelatedPostId']

        for link in link_set:
            if post_id in link:
                link.add(r_post_id)
                break
            elif r_post_id in link:
                link.add(post_id)
                break
        else:
            l = [post_id, r_post_id]
            link_set.append(set(l))

    if check:
        for k in link_set:
            if len(k) > 1:
                print '%s' % str(k)

    print 'Link Sets: %d' % len(link_set)
    return link_set


def merge_linked_qa(qa_pairs, link_set):
    print '\nMERGE LINKED QA'
    merged_qa_pairs = []

    for links in link_set:
        linked_qa = QA()

        for pair in qa_pairs:
            q = pair.questions[0]
            a = pair.answers

            if q.id in links:
                linked_qa.questions.append(q)
                linked_qa.answers.extend(a)
                pair.is_link = True

        if len(linked_qa.questions) > 0:
            merged_qa_pairs.append(linked_qa)

#    for pair in qa_pairs:
#        if pair.is_link is False:
#            merged_qa_pairs.append(pair)

    print 'Merged QA Pairs: %d' % len(merged_qa_pairs)
    return merged_qa_pairs


def main(argv):
    print '\nDATA CREATION START\n'

    posts = load(argv.posts)
    links = load(argv.links)
    link_set = get_link_set(links, argv.check)

    q, a = separate_qa(posts)
    qa_pairs = get_qa_pairs(q, a)
    merged_qa_pairs = merge_linked_qa(qa_pairs, link_set)

    save(fn='test', data=merged_qa_pairs)

