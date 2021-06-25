import os
import urllib.request
import zipfile
from _commons import warn, error, create_dir_path
import numpy as np

URL_PREFIX = 'http://files.grouplens.org/datasets/movielens/'
VARIANTS = ['ml-100k', 'ml-1m', 'ml-10m', 'ml-20m']
FILENAME_SUFFIX = '.zip'

DATA_DIR = 'data'
DATASET_NAME = 'movielens'
ENCODING = 'utf-8'
STRING_ENCODING = 'ISO-8859-1'

"""
u.data -- The full u data set, 100000 ratings by 943 users on 1682 items. Each user has rated at least 20 movies. Users and items are numbered consecutively from 1. The data is randomly ordered. This is a tab separated list of user id | item id | rating | timestamp. The time stamps are unix seconds since 1/1/1970 UTC

u.info -- The number of users, items, and ratings in the u data set.

u.item -- Information about the items (movies); this is a tab separated list of movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western | The last 19 fields are the genres, a 1 indicates the movie is of that genre, a 0 indicates it is not; movies can be in several genres at once. The movie ids are the ones used in the u.data data set.

u.genre -- A list of the genres.

u.user -- Demographic information about the users; this is a tab separated list of user id | age | gender | occupation | zip code The user ids are the ones used in the u.data data set.

u.occupation -- A list of the occupations.
"""

class MovieLens:
    def __init__(self, variant='ml-100k',
                 pos_rating_threshold=4,
                 data_augmentation_mode='binary_unknown'):

        np.random.seed(0)
        self.DATA_AUGMENTATION_MODES = ['binary', 'binary_unknown', 'original']
        self.variant = variant
        self.UNKNOWN_RATING_VAL = 0
        self.POSITIVE_RATING_VAL = 1
        self.NEGATIVE_RATING_VAL = -1
        self.data_augmentation_mode = data_augmentation_mode
        self.pos_rating_threshold = pos_rating_threshold  # positive rating threshold

        self._maybe_download_and_extract()
        self.num_users, self.num_items, self.num_ratings = self._get_num_users_items()
        self.genre_names = self._get_genre_names()
        self.num_genres = len(self.genre_names)
        self.orig_R, self.implicit_ratings = self._get_rating_matrix()
        self.R, self.R_mask = self._augment_R(mode=data_augmentation_mode)
        self.item_titles, self.item_genres = self._get_item_info()

        self.current_user_idx = 0  # How many users have I already returned in get_next_user()
        self.user_indexes = np.array(range(self.R.shape[0]))  # order of selection of users to recommend to
        np.random.shuffle(self.user_indexes)  # iterate through users randomly when selecting the next user

        self.arm_feature_dim = self.get_arm_feature_dim()
        print('self.R的统计数据：')
        self.get_statistics()

    def _augment_R(self, mode):
        R = np.copy(self.orig_R)
        unknown_rating = 0
        if mode == 'binary':
            R[R < self.pos_rating_threshold] = self.UNKNOWN_RATING_VAL
            R[R >= self.pos_rating_threshold] = self.POSITIVE_RATING_VAL
            print("Binarized rating matrix. Ratings < {} turned to {}.".format(self.pos_rating_threshold,
                                                                               self.UNKNOWN_RATING_VAL))
        elif mode == 'binary_unknown':
            unknown_rating = self.UNKNOWN_RATING_VAL
            R[R == 0] = 999
            R[R < self.pos_rating_threshold] = self.NEGATIVE_RATING_VAL
            R[R == 999] = self.UNKNOWN_RATING_VAL
            R[R >= self.pos_rating_threshold] = self.POSITIVE_RATING_VAL
            print("正面的评分 (>={}) 记为 {}, 负面的评分记为 {}, 不确定的评分记为 {}"
                  .format(self.pos_rating_threshold, self.POSITIVE_RATING_VAL, self.NEGATIVE_RATING_VAL,
                          self.UNKNOWN_RATING_VAL))
        elif mode == 'original':
            pass
        else:
            error("ERROR: _augment_R(mode): mode = '{}' is not recognized!".format(mode))
            #print("R will not be modified!")

        R_mask = R != unknown_rating
        self.UNKNOWN_RATING_VAL = unknown_rating
        return R, R_mask

    def add_random_ratings(self, num_to_each_user=10):
        no_items = self.R.shape[1]
        no_users = self.R.shape[0]
        for u in range(no_users):
            ids = np.random.randint(no_items, size=num_to_each_user)
            new_ratings = np.random.randint(2, size=num_to_each_user) * 2 - np.ones(shape=(num_to_each_user,),
                                                                                    dtype=int)
            self.R[u][ids] = new_ratings
            # print('ids:', ids)
            # print('ratings:', ratings)
            # print('R[u]:', self.R[u])
        return self.R

    def recommend(self, user_id, item_id, fixed_rewards=True, prob_reward_p=0.9):
        MIN_PROBABILITY = 0 # Minimal probability to like an item - adds stochasticity

        if self.R[user_id, item_id] == self.POSITIVE_RATING_VAL:
            if fixed_rewards:
                return 1
            else:
                return np.random.binomial(n=1, p=prob_reward_p)  # Bernoulli coin toss
        elif self.R[user_id, item_id] == self.NEGATIVE_RATING_VAL:
            if fixed_rewards:
                return 0
            else:
                return np.random.binomial(n=1, p=1-prob_reward_p)  # Bernoulli coin toss
        else:
            item_genres = self.item_genres[item_id]
            user_ratings = self.R[user_id]
            user_pos_rat_idxs = np.argwhere(user_ratings == self.POSITIVE_RATING_VAL).flatten()
            user_neg_rat_idxs = np.argwhere(user_ratings == self.NEGATIVE_RATING_VAL).flatten()
            num_known_ratings = len(user_pos_rat_idxs) + len(user_neg_rat_idxs)
            genre_idxs = np.argwhere(item_genres == 1).flatten()

            # Find how much user likes the genre of the recommended movie based on his previous ratings.
            genre_likabilities = []
            for genre_idx in genre_idxs:
                genre_likability = 0
                for item_idx in user_pos_rat_idxs:
                    genre_likability += self.item_genres[item_idx][genre_idx]
                for item_idx in user_neg_rat_idxs:
                    genre_likability -= self.item_genres[item_idx][genre_idx]
                genre_likability /= num_known_ratings
                genre_likabilities.append(genre_likability)

            genre_likabilities = np.array(genre_likabilities)

            # how much user user_id likes the genre of the recommended item item_id
            result_genre_likability = np.average(genre_likabilities)
            binomial_reward_probability = result_genre_likability
            if binomial_reward_probability <= 0:
                #print("User={}, item={}, genre likability={}".format(user_id, item_id, result_genre_likability))
                binomial_reward_probability = MIN_PROBABILITY # this could be replaced by small probability

            approx_rating = np.random.binomial(n=1, p=binomial_reward_probability)  # Bernoulli coin toss

            if approx_rating == 1:
                self.R[user_id, item_id] = self.POSITIVE_RATING_VAL
            else:
                self.R[user_id, item_id] = self.NEGATIVE_RATING_VAL

            #return approx_rating
            return approx_rating

    def get_features_of_current_arms(self, t):
        t = t % self.num_users
        user_features = self.R[t]  
        user_features = np.tile(user_features, (self.num_items, 1))  # matrix where each row is R[t]
        item_features = self.item_genres  # matrix
        # 展现的是用户和实例的特征相加
        # arm_feature_dims = item_features.shape[1] + user_features.shape[0]
        arm_features = np.concatenate((user_features, item_features), axis=1)
        return arm_features

    def get_arm_feature_dim(self):
        return self.item_genres.shape[1] + self.R.shape[1]

    def get_uknown_items_of_user(self, user_id):
        user_ratings = self.R[user_id]  
        unknown_item_ids = np.argwhere(user_ratings == self.UNKNOWN_RATING_VAL).flatten()
        return unknown_item_ids

    def get_next_user(self):
        if self.current_user_idx == self.R.shape[0]:
            self.current_user_idx = 0
            np.random.shuffle(self.user_indexes)

        next_user_id = self.user_indexes[self.current_user_idx]
        self.current_user_idx += 1
        return next_user_id

    def shrink(self, max_items):
        num_users = self.R.shape[0]
        num_items = self.R.shape[1]
        if max_items >= num_items:
            warn("movielens.shrink() 最大实例数={} 大于总实例数 = {} => 啥也干不了".format(
                max_items, num_items))

        shrink_ratio = num_items / max_items
        max_users = int(num_users / shrink_ratio)

        self.R = self.R[0:max_users, 0:max_items]
        self.R_mask = self.R_mask[0:max_users, 0:max_items]
        self.num_users = self.R.shape[0]
        self.num_items = self.R.shape[1]
        self.item_genres = self.item_genres[0:self.num_items]
        self.item_titles = self.item_titles[0:self.num_items]
        self.user_indexes = np.array(range(self.R.shape[0]))  
        np.random.shuffle(self.user_indexes)  

        self.arm_feature_dim = self.get_arm_feature_dim()

        print("Shrinked rating matrix from {} to {}.".format(self.orig_R.shape, self.R.shape))
        print("\nAfter shrinking:")
        self.get_statistics()

    def get_statistics(self):
        R = self.R
        total_rats = R.size
        no_rats = len(R[R != self.UNKNOWN_RATING_VAL])
        no_pos_rats = len(R[R == self.POSITIVE_RATING_VAL])
        no_neg_rats = len(R[R == self.NEGATIVE_RATING_VAL])

        user_pos_rats = np.zeros(shape=(R.shape[0],), dtype=int)
        user_neg_rats = np.zeros(shape=(R.shape[0],), dtype=int)
        for u in range(R.shape[0]):
            user = R[u]
            user_pos_rats[u] = len(user[user == self.POSITIVE_RATING_VAL])
            user_neg_rats[u] = len(user[user == self.NEGATIVE_RATING_VAL])

        user_pos_rats_avg = np.average(user_pos_rats)
        user_neg_rats_avg = np.average(user_neg_rats)
        user_pos_rats_std = np.std(user_pos_rats)
        user_neg_rats_std = np.std(user_neg_rats)

        print('用户数量:                ', R.shape[0])
        print('实例数量:                ', R.shape[1])
        print('评分条数:                ', total_rats)
        print('已知评分:                ', no_rats)
        print('已知正面评分:            ', no_pos_rats)
        print('已知负面评分:            ', no_neg_rats)
        print('已知评分占比:            ', no_rats / total_rats)
        print('已知正面评分占比:        ', no_pos_rats / total_rats)
        print('已知负面评分占比:        ', no_neg_rats / total_rats)
        print('用户正面评价均值: {} +- {}'.format(user_pos_rats_avg, user_pos_rats_std))
        print('用户负面评价均值: {} +- {}'.format(user_neg_rats_avg, user_neg_rats_std))
        return (user_pos_rats, user_neg_rats)

    def _maybe_download_and_extract(self):
        if self.variant not in VARIANTS:
            error(
                'ERROR: maybe_download_and_extract(): Provided variant {} is not in {}!'.format(self.variant, VARIANTS))

        filename = self.variant + FILENAME_SUFFIX
        data_dir = os.path.join(DATA_DIR, DATASET_NAME)
        filepath = os.path.join(data_dir, filename)
		
        if not os.path.exists(filepath):
            create_dir_path(data_dir)
            url = URL_PREFIX + filename
            print("Downloading {} from {}".format(filename, url))
			#urllib.request.urlretrieve(url, filepath)
            print("Successfully downloaded {} from {}".format(filename, url))

            print("Extracting {}".format(filename))
            zip_ref = zipfile.ZipFile(filepath, 'r')
            zip_ref.extractall(data_dir)
            zip_ref.close()
            print("Successfully extracted {}".format(filename))
        else:
            print("{} is already downloaded.".format(filepath))
		

    def _get_num_users_items(self):
        filepath = os.path.join(DATA_DIR, DATASET_NAME, self.variant, 'u.info')
        with open(filepath, encoding=ENCODING) as f:
            num_users = int(f.readline().split(' ')[0])
            num_items = int(f.readline().split(' ')[0])
            num_ratings = int(f.readline().split(' ')[0])

        return num_users, num_items, num_ratings

    def _get_rating_matrix(self):
        r = np.zeros(shape=(self.num_users, self.num_items), dtype=float)
        implicit_ratings = np.zeros(shape=(self.num_ratings, 4), dtype=int)

        filename = 'u.data'
        filepath = os.path.join(DATA_DIR, DATASET_NAME, self.variant, filename)
        line_idx = 0
        with open(filepath, encoding=ENCODING) as f:
            for line in f:
                chunks = line.split('\t')
                user_id = int(chunks[0]) - 1  
                item_id = int(chunks[1]) - 1
                rating = int(chunks[2])
                timestamp = int(chunks[3])

                implicit_ratings[line_idx] = np.array([user_id, item_id, rating, timestamp])
                r[user_id, item_id] = rating
                line_idx += 1

        print("创建评分矩阵，长宽为={}，并且类型为={}，采自文件 {}.".format(r.shape, r.dtype, filename))
        return r, implicit_ratings

    def _get_item_info(self):
        genres = np.zeros(shape=(self.num_items, self.num_genres), dtype=float)
        titles = np.empty(shape=(self.num_items,), dtype=object)

        filename = 'u.item'
        filepath = os.path.join(DATA_DIR, DATASET_NAME, self.variant, filename)

        with open(filepath, encoding=STRING_ENCODING) as f:
            for line in f:
                chunks = line.split('|')
                movie_id = int(chunks[0]) - 1  
                title = chunks[1]
                titles[movie_id] = title
             
                for i in range(len(chunks) - 5):
                    genres[movie_id, i] = int(chunks[i + 5])

        print("创建电影种类矩阵，长宽为={}，并且类型为={}，采自文件 {}".format(genres.shape, genres.dtype, filename))
        print("创建电影名称矩阵，长宽为={}，并且类型为={}，采自文件 {}".format(titles.shape, titles.dtype, filename))
        return titles, genres

    def _get_genre_names(self):
        filename = 'u.genre'
        filepath = os.path.join(DATA_DIR, DATASET_NAME, self.variant, filename)
        with open(filepath, encoding=ENCODING) as f:
            genres = [x.strip().split('|')[0] for x in f.readlines()]
            genres = [x for x in genres if len(x) > 0]
            
        return genres
