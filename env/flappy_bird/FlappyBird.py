import os
import time

import pygame
import sys


class Bird(object):
    """定义一个鸟类"""

    def __init__(self):
        """定义初始化方法"""

        current_path = os.path.abspath(__file__)
        dir = os.path.dirname(current_path)

        self.birdRect = pygame.Rect(65, 350, 50, 50)  # 鸟的矩形
        # 定义鸟的3种状态列表
        self.birdStatus = [pygame.image.load(f"{dir}/assets/1.png"),
                           pygame.image.load(f"{dir}/assets/2.png"),
                           pygame.image.load(f"{dir}/assets/dead.png")]
        self.status = 0      # 默认飞行状态
        self.birdX = 120     # 鸟所在X轴坐标,即是向右飞行的速度
        self.birdY = 350     # 鸟所在Y轴坐标,即上下飞行高度
        self.jump = False    # 默认情况小鸟自动降落
        self.jumpSpeed = 10  # 跳跃高度
        self.gravity = 5     # 重力
        self.dead = False    # 默认小鸟生命状态为活着

    def reset(self):
        self.birdRect = pygame.Rect(65, 350, 50, 50)  # 鸟的矩形

        self.status = 0  # 默认飞行状态
        self.birdX = 120  # 鸟所在X轴坐标,即是向右飞行的速度
        self.birdY = 350  # 鸟所在Y轴坐标,即上下飞行高度
        self.jump = False  # 默认情况小鸟自动降落
        self.jumpSpeed = 10  # 跳跃高度
        self.gravity = 5  # 重力
        self.dead = False  # 默认小鸟生命状态为活着

    def birdUpdate(self):
        if self.jump:
            # 小鸟跳跃
            self.jumpSpeed -= 1           # 速度递减，上升越来越慢
            self.birdY -= self.jumpSpeed  # 鸟Y轴坐标减小，小鸟上升
        else:
            # 小鸟坠落
            self.gravity += 0.2           # 重力递增，下降越来越快
            self.birdY += self.gravity    # 鸟Y轴坐标增加，小鸟下降
        self.birdRect[1] = self.birdY     # 更改Y轴位置


class Pipeline(object):
    """定义一个管道类"""

    def __init__(self):
        """定义初始化方法"""
        self.score = 0
        self.wallx = 400  # 管道所在X轴坐标

        current_path = os.path.abspath(__file__)
        dir = os.path.dirname(current_path)

        self.pineUp = pygame.image.load(f"{dir}/assets/top.png")
        self.pineDown = pygame.image.load(f"{dir}/assets/bottom.png")

    def reset(self):
        self.wallx = 400  # 管道所在X轴坐标
        self.score = 0

    def updatePipeline(self):
        """"管道移动方法"""

        reward = 0
        self.wallx -= 5  # 管道X轴坐标递减，即管道向左移动
        # 当管道运行到一定位置，即小鸟飞越管道，分数加1，并且重置管道
        if self.wallx < -40:
            reward = 100
            self.score += 1
            self.wallx = 400
        return reward

class FlappyBird():
    def __init__(self):

        self.action_space = ['f', 'w'] # flay, wait
        self.n_actions = len(self.action_space)
        self.n_features = 3

        pygame.init()  # 初始化pygame
        pygame.font.init()  # 初始化字体

        current_path = os.path.abspath(__file__)
        dir = os.path.dirname(current_path)
        self.background = pygame.image.load(f"{dir}/assets/background.png")
        self.font = pygame.font.SysFont("Arial", 50)  # 设置字体和大小

        self.size = self.width, self.height = 400, 650  # 设置窗口

        self.screen = pygame.display.set_mode(self.size)  # 显示窗口
        self.clock = pygame.time.Clock()  # 设置时钟
        self.pipeLine = Pipeline()  # 实例化管道类
        self.bird = Bird()  # 实例化鸟类

    def createMap(self):
        """定义创建地图的方法"""
        self.screen.fill((255, 255, 255))  # 填充颜色
        self.screen.blit(self.background, (0, 0))  # 填入到背景

        # 显示管道
        self.screen.blit(self.pipeLine.pineUp, (self.pipeLine.wallx, -300))  # 上管道坐标位置
        self.screen.blit(self.pipeLine.pineDown, (self.pipeLine.wallx, 500))  # 下管道坐标位置
        reward = self.pipeLine.updatePipeline()  # 管道移动

        # 显示小鸟
        if self.bird.dead:  # 撞管道状态
            self.bird.status = 2
        elif self.bird.jump:  # 起飞状态
            self.bird.status = 1
        self.screen.blit(self.bird.birdStatus[self.bird.status], (self.bird.birdX, self.bird.birdY))  # 设置小鸟的坐标
        self.bird.birdUpdate()  # 鸟移动

        self.bird.status = 0


        # 显示分数
        self.screen.blit(self.font.render('Score:' + str(self.pipeLine.score), -1, (255, 255, 255)), (100, 50))  # 设置颜色及坐标位置
        pygame.display.update()  # 更新显示
        return reward

    def checkDead(self):
        # 上方管子的矩形位置
        upRect = pygame.Rect(self.pipeLine.wallx, -300,
                             self.pipeLine.pineUp.get_width() - 10,
                             self.pipeLine.pineUp.get_height())

        # 下方管子的矩形位置
        downRect = pygame.Rect(self.pipeLine.wallx, 500,
                               self.pipeLine.pineDown.get_width() - 10,
                               self.pipeLine.pineDown.get_height())
        # 检测小鸟与上下方管子是否碰撞
        if upRect.colliderect(self.bird.birdRect) or downRect.colliderect(self.bird.birdRect):
            self.bird.dead = True

        state_ = [upRect[0] - (self.bird.birdRect[0] + self.bird.birdRect[2]),
                         self.bird.birdRect[1] - (upRect[3] + upRect[1]),
                         downRect[1] - (self.bird.birdRect[1] + self.bird.birdRect[3])]

        # 检测小鸟是否飞出上下边界
        if not 0 < self.bird.birdRect[1] < self.height:
            self.bird.dead = True
            return state_, True, 'dead'
        else:
            return state_, False, 'live'

    def getResutl(self):
        final_text1 = "Game Over"
        final_text2 = "Your final score is:  " + str(self.pipeLine.score)
        ft1_font = pygame.font.SysFont("Arial", 70)  # 设置第一行文字字体
        ft1_surf = self.font.render(final_text1, 1, (242, 3, 36))  # 设置第一行文字颜色
        ft2_font = pygame.font.SysFont("Arial", 50)  # 设置第二行文字字体
        ft2_surf = self.font.render(final_text2, 1, (253, 177, 6))  # 设置第二行文字颜色
        self.screen.blit(ft1_surf, [self.screen.get_width() / 2 - ft1_surf.get_width() / 2, 100])  # 设置第一行文字显示位置
        self.screen.blit(ft2_surf, [self.screen.get_width() / 2 - ft2_surf.get_width() / 2, 200])  # 设置第二行文字显示位置
        pygame.display.flip()  # 更新整个待显示的Surface对象到屏幕上

    def step(self, action):
        if action == 0 and not self.bird.dead:
            self.bird.jump = True  # 跳跃
            self.bird.gravity = 5  # 重力
            self.bird.jumpSpeed = 10  # 跳跃速度
        else:
            self.bird.jump = False  # 跳跃

        reward = self.createMap()

        state_, dead, info = self.checkDead()

        if dead:
            reward = -100
            self.getResutl()  # 如果小鸟死亡，显示游戏总分数

        return state_, reward, dead, info



    def reset(self):
        time.sleep(2)
        self.bird.reset()
        self.pipeLine.reset()

        # 上方管子的矩形位置
        upRect = pygame.Rect(self.pipeLine.wallx, -300,
                             self.pipeLine.pineUp.get_width() - 10,
                             self.pipeLine.pineUp.get_height())

        # 下方管子的矩形位置
        downRect = pygame.Rect(self.pipeLine.wallx, 500,
                               self.pipeLine.pineDown.get_width() - 10,
                               self.pipeLine.pineDown.get_height())
        state_ = [upRect[0] - (self.bird.birdRect[0] + self.bird.birdRect[2]),
                  self.bird.birdRect[1] - (upRect[3] + upRect[1]),
                  downRect[1] - (self.bird.birdRect[1] + self.bird.birdRect[3])]
        return state_

    def destroy(self):
        pygame.quit()

