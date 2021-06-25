from _commons import warn, error, create_dir_path
import numpy as np
import time
import math
import random
from movielens import MovieLens


class LinUCB:
    def __init__(self, alpha, dataset=None, max_items=500, allow_selecting_known_arms=True, fixed_rewards=True,
                 prob_reward_p=0.9):
        if dataset is None:
            self.dataset = MovieLens(variant='ml-100k',
                                     pos_rating_threshold=4,
                                     data_augmentation_mode='binary_unknown')
        else:
            self.dataset = dataset
        self.dataset.shrink(max_items)
        self.dataset.add_random_ratings(num_to_each_user=3)
        self.alpha = alpha
        self.fixed_rewards = fixed_rewards
        self.prob_reward_p = prob_reward_p
        self.users_with_unrated_items = np.array(range(self.dataset.num_users))
        self.monitored_user = np.random.choice(self.users_with_unrated_items)
        self.allow_selecting_known_arms = allow_selecting_known_arms
        self.d = self.dataset.arm_feature_dim
        self.b = np.zeros(shape=(self.dataset.num_items, self.d))

        
        print("\n初始化矩阵 A {} 并且消耗了 {}MB 的内存"
              .format((self.dataset.num_items, self.d, self.d), 8 * self.dataset.num_items * self.d * self.d / 1e6))
        self.A = np.repeat(np.identity(self.d, dtype=float)[
                           np.newaxis, :, :], self.dataset.num_items, axis=0)
        print("\n成功初始化LinUCB")



    def choose_arm(self, t, unknown_item_ids, verbosity, epoch):
        A = self.A
        b = self.b
        arm_features = self.dataset.get_features_of_current_arms(t=t)
        p_t = np.zeros(shape=(arm_features.shape[0],), dtype=float)
        p_t -= 9999  
        item_ids = unknown_item_ids

        if self.allow_selecting_known_arms:
            item_ids = range(self.dataset.num_items)
            p_t += 9999

        for a in item_ids:  
            x_ta = arm_features[a].reshape(
                arm_features[a].shape[0], 1)  
            A_a_inv = np.linalg.inv(A[a])
            theta_a = A_a_inv.dot(b[a])
            p_t[a] = theta_a.T.dot(x_ta) + self.alpha * \
                np.sqrt(x_ta.T.dot(A_a_inv).dot(x_ta))
        # 就在这里改，改成前K的值，不在只选取最优解，看看效果
        # max_p_t变成一连串的值
        # 再用模拟退火来约束
        max_p_t = np.max(p_t)
        post_energy = (-1)*(1/(1 + math.e ** (-10*(epoch+1))))
        head_energy = (-1)*(1/(1 + math.e ** (-10*(epoch))))

        energy_change = post_energy - head_energy

        P = math.exp(energy_change)

        top_k = 3
        t_set = set(p_t)
        t_list = list(t_set)
        t_list.sort()
        t_list.reverse()
        sorted_t_list = t_list
        top_k_p_t = sorted_t_list[:top_k]

        #naive_p_t = np.random.choice(top_k_p_t)
        tag = 0
        #P = 0.99
        while tag == 0:
            naive_p_t = np.random.choice(top_k_p_t)
            if max_p_t == naive_p_t:
                max_idxs = np.argwhere(p_t == naive_p_t).flatten()
                tag = 1
            else:
                pp = random.randint(0, 100)/100
                if pp >= P:
                    max_idxs = np.argwhere(p_t == naive_p_t).flatten()
                    tag = 1
                else:
                    tag = 0
					
		# 被注释掉的是naive的LinUCB算法
        # max_idxs = np.argwhere(p_t == max_p_t).flatten()
        if max_p_t <= 0:
            print("用户 {} 的最大 p_t={}, p_t={}".format(t, max_p_t, p_t))

        # max_idxs = np.argwhere(p_t == max_p_t).flatten()
        # idx of article to recommend to user t
        a_t = np.random.choice(max_idxs)

        # observed reward = 1/0
        r_t = self.dataset.recommend(user_id=t, item_id=a_t,
                                     fixed_rewards=self.fixed_rewards, prob_reward_p=self.prob_reward_p)

        if verbosity >= 2:
            print("用户 {} 选择了实例 {} ，评分为 p_t={} ，奖励为 {}".format(
                t, a_t, p_t[a_t], r_t))

        x_t_at = arm_features[a_t].reshape(
            arm_features[a_t].shape[0], 1)  
        A[a_t] = A[a_t] + x_t_at.dot(x_t_at.T)
        
        b[a_t] = b[a_t] + r_t * x_t_at.flatten()

        return r_t

    def run_epoch(self, verbosity=2):
        """
        Call choose_arm() for each user in the dataset.
        :return: Average received reward.
        """
        rewards = []
        start_time = time.time()

        for i in range(self.dataset.num_users):
            start_time_i = time.time()
            user_id = self.dataset.get_next_user()
            # user_id = 1
            unknown_item_ids = self.dataset.get_uknown_items_of_user(user_id)

            if self.allow_selecting_known_arms == False:
                if user_id not in self.users_with_unrated_items:
                    continue

                if unknown_item_ids.size == 0:
                    print(
                        "用户 {} 没有未知的评分实例, 跳过该用户".format(user_id))
                    self.users_with_unrated_items = self.users_with_unrated_items[
                        self.users_with_unrated_items != user_id]
                    continue

            rewards.append(self.choose_arm(
                user_id, unknown_item_ids, verbosity, i))
            time_i = time.time() - start_time_i
            if verbosity >= 2:
                print("Choosing arm for user {}/{} ended with reward {} in {}s".format(i, self.dataset.num_users,
                                                                                       rewards[i], time_i))

        total_time = time.time() - start_time
        avg_reward = np.average(np.array(rewards))
        return avg_reward, total_time

    def run(self, num_epochs, verbosity=1):
        self.users_with_unrated_items = np.array(range(self.dataset.num_users))
        avg_rewards = np.zeros(shape=(num_epochs,), dtype=float)
        for i in range(num_epochs):
            avg_rewards[i], total_time = self.run_epoch(verbosity)

            if verbosity >= 1:
                print(
                    "完成 epoch {}/{} ，平均奖励为 {} ，运行时间为 {}s".format(i, num_epochs, avg_rewards[i], total_time))
        return avg_rewards
