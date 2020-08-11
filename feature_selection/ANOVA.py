#
# # coding:utf-8
# """
# Author : LinDing Group (ZHAO)
# Application : feature selection
# Date : 20160825
# Usage: Please refer to the README file
# Intro : This is a feature selection technology based on the analysis of variance!
# （源码略有改动）
# """
#
# def generate_ANONA_sorted_file(csvfile, outputfile):
#     [F_values, group_num, sample_num, total_sample, feature_num] = obtain_ANONA_values_dict(csvfile)
#     F_values = sorted(F_values.items(), key=lambda asd: asd[1], reverse=True)
#
#     g = open(outputfile, 'w')
#     g.write(get_output_prompt(group_num, sample_num, total_sample, feature_num))
#     g.write('-------ANONA Sorted Feature Set-------\n')
#     g.write('Rank\tfeature\tvalue\n')
#     for i in range(len(F_values)):
#         g.write('%d\t%s\t%f\n' % (i + 1, F_values[i][0], F_values[i][1]))
#
#     g.close()
#     print('Successful!')
#
# def  generate_ANONA_sorted_feature_list(csvfile):
#     [F_values, group_num, sample_num, total_sample, feature_num] = obtain_ANONA_values_dict(csvfile)
#     F_list=[]
#     for x in F_values.items():
#         if x[1] != 0:
#             F_list.append(x)
#
#     F_values = sorted(F_list, key=lambda asd: asd[1], reverse=True)
#
#     feature_list=[]
#     for i in range(len(F_values)):
#         #g.write('%d\t%s\t%f\n' % (i + 1, F_values[i][0], F_values[i][1]))
#         feature_list.append(F_values[i][0])
#     return feature_list
#
# def obtain_ANONA_values_dict(filename):
#     [note_label, key_words, csv_dics] = csv_to_dict(filename)
#     feature_num = len(note_label)
#
#     F_values = dict()
#     [group_num, sample_num, total_sample] = get_sample_num(key_words)
#
#     for label_index in range(len(note_label)):
#         label = note_label[label_index]
#
#         assign_label_values = get_assign_label_values(csv_dics, label_index)
#         if not label:
#             continue
#         F_values[label] = calculate_thisLabel_Fvalue(assign_label_values, group_num, sample_num, total_sample)
#
#     return F_values, group_num, sample_num, total_sample, feature_num
#
#
# def csv_to_dict(filename):
#     csv_dics = dict()
#     key_words = []
#
#     f = open(filename).readlines()
#     note_label = get_csv_note_label(f[0])
#     f = f[1:]
#     for line_index in range(len(f)):
#         line = f[line_index].strip().split(',')
#         key_word = line[0]
#         key_value = line[1:]
#
#         key_words.append(key_word)
#         if csv_dics.get(key_word) == None:
#             csv_dics[key_word] = []
#         else:
#             pass
#         csv_dics[key_word].append(key_value)
#
#     return note_label, key_words, csv_dics
#
# def get_csv_note_label(seq):
#     seq = seq.strip().split(',')
#     note_label = seq[1:]
#     return note_label
#
# def get_sample_num(key_words):
#     sample_num = dict()
#     total_sample = 0
#
#     key_words_set = set(key_words)
#     group_num = len(key_words_set)
#     for each in key_words_set:
#         assign_sample_num = key_words.count(each)
#         sample_num[each] = assign_sample_num
#         total_sample += assign_sample_num
#
#     return group_num, sample_num, total_sample
#
#
# def get_assign_label_values(csv_dics, label_index):
#     assign_label_values = dict()
#
#     for each_class in csv_dics.keys():
#         content = []
#         key_values = csv_dics[each_class]
#         for each_values_index in range(len(key_values)):
#             s=key_values[each_values_index][label_index]
#             if not s:
#                 continue
#             content.append(float(key_values[each_values_index][label_index]))
#         assign_label_values[each_class] = content
#
#     return assign_label_values
#
#
# def calculate_thisLabel_Fvalue(assign_label_values, group_num, sample_num, total_sample):
#     [within_group_mean, total_mean] = calculate_withinGroup_and_total_mean(assign_label_values, sample_num,
#                                                                  total_sample)
#
#
#     SSb = 0
#     for each_assign_label in within_group_mean.keys():
#         SSb += (((within_group_mean[each_assign_label] - total_mean) ** 2) * sample_num[each_assign_label])
#
#     SSw = calculate_SumOfSquaresWithinGroups(assign_label_values, within_group_mean)
#     if SSw==0:
#         return 0
#     else:
#         Fvalue = (SSb / (group_num - 1)) / (SSw / (total_sample - group_num))
#         return Fvalue
#
#
# def calculate_withinGroup_and_total_mean(assign_label_values, sample_num, total_sample):
#     within_group_mean = dict()
#     summary = 0
#
#     for each_group_label in assign_label_values.keys():
#         each_group_summary = sum(assign_label_values[each_group_label])
#
#         within_group_mean[each_group_label] = each_group_summary / sample_num[each_group_label]
#         summary += each_group_summary
#
#     total_mean = summary / total_sample
#
#     return within_group_mean, total_mean
#
#
# def calculate_SumOfSquaresWithinGroups(assign_label_values, within_group_mean):
#     value = 0
#
#     for each_assign_label in assign_label_values.keys():
#         for each_value in assign_label_values[each_assign_label]:
#             value += ((each_value - within_group_mean[each_assign_label]) ** 2)
#
#     return value
#
#
# def get_output_prompt(group_num, sample_num, total_sample, feature_num):
#     content = 'There is a %d classification problem!\n\tTotal sample numbers : %d\n\tTotal feature numbers : %d\n\n' \
#               % (group_num, total_sample, feature_num)
#
#     for each in sample_num.keys():
#         content += '\t\tThe sample numbers of label \'%s\' : %d\n' % (each, sample_num[each])
#     content += '\n'
#
#     return content
#
# def run(csvfile,logger):
#
#     #generate_ANONA_sorted_file(csvfile, outputfile)
#     logger.info('ANOVA start...')
#     feature_list=generate_ANONA_sorted_feature_list(csvfile)
#
#     logger.info('ANOVA end.')
#
#     return feature_list
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest,f_classif
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def anova(file):

    dataset = pd.read_csv(file,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model1 = SelectKBest(f_classif, k=1)  # 选择k个最佳特征
    model1.fit_transform(X,y)
    result = [(x,y) for x,y in zip(features_name[1:],model1.scores_)]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [x[0] for x in result]

def run(csvfile,logger):
    logger.info('ANOVA start...')
    feature_list = anova(csvfile)
    logger.info('ANOVA  end.')
    return feature_list
if __name__ == '__main__':

    run('TPC.csv',1)
