### Based on FW notebook and method of moving averages.
### Similarity model added by GN after the moving averages part
### So if the Type Feature Score is the same, the items are ranked by similarity between keywords

## PLEASE NOTE

## Word list is appended randomly, to avoid multiple lists per activity - needs to be sorted later depending on guidance

## For now it works with manually inputting the query, next step is to look at the keywords of the last activity completed

# Working on top of Yih_Dar_SHIEH's work https://colab.research.google.com/drive/10R6lfbUPFHeytGRWPqHRwns2pFjzR-1r
# Extended by Fubao Wu

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import linalg
import matplotlib
import seaborn as sns

#Word Based Inputs
import string
import scipy
from sentence_transformers import SentenceTransformer

def content_based_recommendation(user_id, queries, closest_n=10, activity_used_feature_type = 'Content_type',
        alpha = 0.9, iterate_count=1, MAX_ITER_NUM=2, step_size=5):

    percentile_values=step_size

    # read tables
    # because of data generated random with different table, one table has activity_score, the other table has timestamp, so we read two tables now
    # in the future, there should be one student_actviity_table
    student_activity_table_v1 = pd.read_csv("student_activity_table_v1.csv", index_col=0)
    student_activity_table_v2 = pd.read_csv("student_activity_v2.csv", index_col=0)
    activities_encode = pd.read_csv("activities_encoded.csv", index_col=0)

    # merge two student activity tables
    student_activity_table = student_activity_table_v1.merge(student_activity_table_v2, on='User_ID', how='inner')
    # remove one Activity_ID_y,  and keep Activity_ID_x for test
    student_activity_table.drop(['Activity_ID_y'], axis=1, inplace=True)
    # change Activity_ID_x to Activity_ID_y
    student_activity_table.rename(columns={'Activity_ID_x':'Activity_ID'}, inplace=True)

    # Also mapping them to numerical IDs
    game_type_vocab = sorted(list({k for k in activities_encode["GameType"]}))
    game_type_to_id = {v: k for k, v in enumerate(game_type_vocab)}

    content_type_vocab = sorted(list({k for k in activities_encode["Activity_Content"]}))
    content_type_to_id = {v: k for k, v in enumerate(content_type_vocab)}

    # make sure `Activity_ID` in `activities_encode` is unique
    assert len(activities_encode) == len(activities_encode["Activity_ID"].unique())
    encoded_id_to_activity_id_dict = activities_encode["Activity_ID"].to_dict()
    activity_id_to_encoded_id_dict = {v: k for k, v in encoded_id_to_activity_id_dict.items()}  #inverted

    # Add mapped ID
    activities_encode['Activity_Num_ID'] = activities_encode['Activity_ID'].map(activity_id_to_encoded_id_dict)

    activity_max = len(activities_encode)

    activities_encode['Game_Type_ID'] = activities_encode["GameType"].map(game_type_to_id)
    activities_encode['Content_Type_ID'] = activities_encode["Activity_Content"].map(content_type_to_id)

    game_type_id_max = len(game_type_vocab)
    content_type_id_max = len(content_type_vocab)

    sparse_game_type_feature_tensor = tf.SparseTensor(
        indices=activities_encode[['Activity_Num_ID', 'Game_Type_ID']].values,
        values=tf.ones(shape=(len(activities_encode),), dtype=tf.float32),
        dense_shape= (activity_max, game_type_id_max)
    )

    sparse_content_type_feature_tensor = tf.SparseTensor(
        indices=activities_encode[['Activity_Num_ID', 'Content_Type_ID']].values,
        values=tf.ones(shape=(len(activities_encode),), dtype=tf.float32),
        dense_shape= (activity_max, content_type_id_max)
    )

    student_activity_table['Activity_Num_ID'] = student_activity_table['Activity_ID'].map(activity_id_to_encoded_id_dict)

    # add target score
    def score(hits, misses, Max_Hits, Complete_time, k = 0.4, w1 = 0.8, w2 = 0.2):
        '''
        Calculates score based on number of hits, misses, max_hits, and completion time
        '''
        if (misses>=Max_Hits):
            s1 = 0
        elif (hits == 0):
            s1 = 0
        else:
            s1 = max(((hits/Max_Hits)-k*(misses/Max_Hits))*100,0)

        '''
        Calculates score based on gameplay time
        '''
        # math.sqrt(1.0*max_hits/max_hits_across) * (weight_GT - (((gameplaytime * gameplaytime)/(Max_GT * Max_GT)) * weight_GT))
        s2 = max(np.sqrt(Max_Hits/10) * (100 - (((Complete_time * Complete_time)/(500 * 500 )) * 100)),0)

        '''
        Calculates activity score
        '''
        ac_score = (s1*w1 + s2*w2)/(w1+w2)

        return ac_score
    student_activity_table['target_score'] = student_activity_table.apply(lambda x: score(x['Hits'], x['Misses'], x['Max_Hits'], x['GamePlayTime']),axis=1)

    # get a timestamp statistics
    #sns.lineplot('TimeStamp', data = student_activity_table)

    student_activity_table['TimeStamp'].describe()
    time_stamps_values = student_activity_table['TimeStamp'].values

    start_time_stamp = min(time_stamps_values)
    #print('start_time_stamp: ', start_time_stamp)

    next_time_stamp = np.percentile(time_stamps_values, 10)

    # get the next time stamp with the step size
    def get_next_time_stamp(time_stamps_values, step_size):
        next_time_stamp = np.percentile(time_stamps_values, step_size)
        return next_time_stamp

    #get the table within the time stamp period
    def get_student_activity_table_period(start_time_stamp, end_time_stamp):
        student_activity_table_period = student_activity_table[(student_activity_table['TimeStamp'] >= start_time_stamp) & (student_activity_table['TimeStamp'] <= end_time_stamp)]
        return student_activity_table_period

    def tf_spare_multiply(a: tf.SparseTensor, b: tf.SparseTensor):
        a_sm = linalg.sparse.sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            a.indices, a.values, a.dense_shape
        )

        b_sm = linalg.sparse.sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            b.indices, b.values, b.dense_shape
        )

        c_sm = linalg.sparse.sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
            a=a_sm, b=b_sm, type=tf.float32
        )

        c = linalg.sparse.sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
            c_sm, tf.float32
        )

        return tf.SparseTensor(
            c.indices, c.values, dense_shape=c.dense_shape
        )

    # get the student activity feature tensorflow matrix
    def get_student_activity_feature(student_activity_table_period, activity_feature_tensor):
        user_max = max(student_activity_table['User_ID'].values) + 1
        sparse_score_tensor = tf.SparseTensor(
            indices=student_activity_table[['User_ID', 'Activity_Num_ID']].values,
            values = tf.constant(student_activity_table['target_score'], dtype=tf.float32),
            dense_shape= (user_max, activity_max),
        )

        # print(sparse_score_tensor)
        user_activity_feature_matrix = tf_spare_multiply(sparse_score_tensor, activity_feature_tensor)
        return user_activity_feature_matrix

    # moving average to calculate student activity feature
    def get_moving_average_student_activity_feature(instant_user_feature_matrix, previous_user_feature_matrix, alpha):
        # update current_user_feature_matrix
        current_user_feature_matrix = instant_user_feature_matrix * alpha + previous_user_feature_matrix * (1-alpha)
        return current_user_feature_matrix

    # get recommendation score for one user
    def get_recommendation_scores_one_user(user_id, user_feature_matrix, activity_feature_matrix):

        user_feature_vector = user_feature_matrix[user_id]
        recommendation_scores = tf.sparse.sparse_dense_matmul(activity_feature_matrix, user_feature_vector[:, None])
        return recommendation_scores[:, 0]
    # Visualizing User / Feature scores when recommending
    def visualize(user_id, feature_type, max_types=10):

        feature_type_mapping = {
            'Game_type': [user_game_feature_matrix, game_type_vocab,],
            #'Content_type': [user_content_feature_matrix, content_type_vocab,],
            #'Combined_type': [user_combined_feature_matrix, combined_type_vocab,],
        }

        user_feature_matrix, feature_vocab = feature_type_mapping[feature_type]

        user_features = pd.DataFrame(user_feature_matrix[user_id], index=feature_vocab, columns=['Feature score'])
        user_features_sorted = user_features.sort_values(by='Feature score', ascending=False)[:max_types]

        ax = user_features_sorted.plot.barh(figsize=(16, 16))
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_title(f"User {user_id} / {feature_type}_feature")
        ax.set_xlabel("Feature score");
        ax.set_ylabel("")

    # moving average to simulate time series
    student_activity_table['TimeStamp'].describe()
    time_stamps_values = student_activity_table['TimeStamp'].values

    start_time_stamp = min(time_stamps_values)
    #print('start_time_stamp: ', start_time_stamp)

    previous_user_feature_matrix = 0
    while(iterate_count < MAX_ITER_NUM and percentile_values <= 100):

      # 1) start a timestampe with at least some activites to get student activities features $f_s$
        next_time_stamp = get_next_time_stamp(time_stamps_values, step_size)
        # get the student activity feature; test with gameType
          # -- get the student activity df frist with the start_time_stamp and
        df_student_activity_current_period = get_student_activity_table_period(start_time_stamp, next_time_stamp)
        #print('df_student_activity_current_period: ', df_student_activity_current_period.head(5))

        if activity_used_feature_type == 'Game_type':

            instant_user_game_feature_matrix = get_student_activity_feature(df_student_activity_current_period, sparse_game_type_feature_tensor)
            instant_user_game_feature_matrix = tf.sparse.to_dense(instant_user_game_feature_matrix)

            # moving average to get feature
            user_game_feature_matrix = get_moving_average_student_activity_feature(instant_user_game_feature_matrix, previous_user_feature_matrix, alpha)
    #        print("user_game_feature_matrix: ", user_game_feature_matrix)

            # normalize
            user_game_feature_matrix = user_game_feature_matrix / tf.reduce_sum(user_game_feature_matrix, axis=1)[:, None]  #[:,None] so broadcasting works properly

            recommendation_scores_one_user = get_recommendation_scores_one_user(user_id, user_game_feature_matrix, sparse_game_type_feature_tensor)

            df_one_user_recommended_scores = pd.DataFrame(recommendation_scores_one_user, columns=['Game_type_feat_score'])
            df_one_user_recommended_scores = pd.concat([activities_encode[['Activity_ID', 'GameType', 'Activity_Content']], df_one_user_recommended_scores], axis=1)


            df_one_user_recommended_scores = df_one_user_recommended_scores.sort_values(by='Game_type_feat_score', ascending=False)


            #print("df_one_user_recommended_scores based on Game type: ", df_one_user_recommended_scores.head(5))
            visualize(user_id=user_id, feature_type='Game_type', max_types=10)
            # update user feature with last value
            previous_user_feature_matrix = user_game_feature_matrix

        elif activity_used_feature_type == 'Content_type':
            instant_user_content_feature_matrix = get_student_activity_feature(df_student_activity_current_period, sparse_content_type_feature_tensor)
            instant_user_content_feature_matrix = tf.sparse.to_dense(instant_user_content_feature_matrix)

            # moving average to get feature
            user_content_feature_matrix = get_moving_average_student_activity_feature(instant_user_content_feature_matrix, previous_user_feature_matrix, alpha)
            #print("user_game_feature_matrix: ", user_content_feature_matrix)

            # normalize
            user_content_feature_matrix = user_content_feature_matrix / tf.reduce_sum(user_content_feature_matrix, axis=1)[:, None]  #[:,None] so broadcasting works properly

            recommendation_scores_one_user = get_recommendation_scores_one_user(user_id, user_content_feature_matrix, sparse_content_type_feature_tensor)

            df_one_user_recommended_scores = pd.DataFrame(recommendation_scores_one_user, columns=['Content_type_feat_score'])
            df_one_user_recommended_scores = pd.concat([activities_encode[['Activity_ID', 'GameType', 'Activity_Content']], df_one_user_recommended_scores], axis=1)
            df_one_user_recommended_scores = df_one_user_recommended_scores.sort_values(by='Content_type_feat_score', ascending=False)

            #print("df_one_user_recommended_scores based on Content type: ", df_one_user_recommended_scores.head(5))
            visualize(user_id=user_id, feature_type='Content_type', max_types=10)
            # update user feature with last value
            previous_user_feature_matrix = user_content_feature_matrix

        iterate_count += 1
        percentile_values = iterate_count*step_size
        start_time_stamp = next_time_stamp

    # append the word list randomly
    df_one_user_recommended_scores = pd.concat([student_activity_table[['Words_List']], df_one_user_recommended_scores], axis=1)
    # remove the NANs resulting from appending randomly
    df_one_user_recommended_scores.dropna(axis=0, inplace=True)

    #sort out the list which is actually not a list but a string
    # get rid of punctuation - use the string library which has punctuation functionn
    df_one_user_recommended_scores["Words_List_clean"]=df_one_user_recommended_scores["Words_List"].str.translate(str.maketrans('','',string.punctuation))
    # if word list empty use the activity content as a list
    df_one_user_recommended_scores["Activity_Words"]=np.where((df_one_user_recommended_scores["Words_List_clean"].str.len()==0),df_one_user_recommended_scores["Activity_Content"].to_list(),df_one_user_recommended_scores["Words_List_clean"])
    # Convert the strings into lists of word lists
    df_one_user_recommended_scores['Activity_Words_List'] = list(map(str,df_one_user_recommended_scores['Activity_Words'].str.split(" ")))

    # Get a vector for each list
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    query_embeddings = model.encode(queries)
    corpus_embeddings = model.encode(df_one_user_recommended_scores['Activity_Words_List'])

    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        #print("\n\n======================\n\n")
        #print("Query:", query)
        #print("\nTop 10 most similar queries in corpus:")

        #for idx, distance in results[0:closest_n]:
        #    print(df_one_user_recommended_scores['Activity_ID'][idx].strip(), df_one_user_recommended_scores['Activity_Words_List'][idx].strip(), "(Score: %.4f)" % (1-distance))

    # record the result back in the table
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        #print("Query:", query)
        for idx, distance in results[0:]:
            df_one_user_recommended_scores['Activity_Similarity_Score']=(1-distance)

    # take top ones with equally high score
    user_recommendations_sorted=df_one_user_recommended_scores.sort_values(by=["Game_type_feat_score","Activity_Similarity_Score"],ascending=[False, False])

    # group by
    user_recommendations_grouped=user_recommendations_sorted.set_index(["Game_type_feat_score","Activity_Similarity_Score"])
    return user_recommendations_sorted, user_recommendations_grouped

if __name__ == "__main__":
    #Moving Average Parameters:
    user_id = 0    # for one user testing here,  could change to multiple users later
    #Word Similiarty Parameters
    queries = [
        'feelings']
    # For each search term return 10 closest matches
    closest_n = 10
    activity_used_feature_type = 'Game_type' # 'Content_type'   # 'Game_type'
    alpha = 0.9  # for moving average parameter, need to tune for better value
    iterate_count = 1
    MAX_ITER_NUM = 2
    step_size = 5

    # Define search queries and embed them to vectors as well
    # get queries manually for now - can be anything, incl freetext

    user_rec_sorted, user_rec_grouped  = content_based_recommendation(user_id, queries, closest_n, activity_used_feature_type, alpha, iterate_count, MAX_ITER_NUM, step_size)
    user_rec_sorted.to_csv('recommendations_sorted.csv')
    user_rec_grouped.to_csv('recommendations_grouped.csv')
