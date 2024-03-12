from manim import *
import random
import math

class Intro(Scene):
    
    def construct(self):
    
        self.camera.background_color = "#353537"
    
        lateral_bar = Rectangle(height = config.frame_height, width = 0.5, fill_color = "#6f6f74", fill_opacity = 1, stroke_width = 0).to_edge(LEFT, buff = 0)
    
        title = Text("Quantum Machine Learning", font_size = 72)
        
        self.play(Write(title), run_time = 3)
        
        self.wait(1)

        trans_title = VGroup(Text("Quantum Machine", font_size = 40), Text("Learning", font_size = 40)).arrange(DOWN, aligned_edge=LEFT)
        trans_title.to_corner(UP + LEFT, buff = 0.8).shift(0.5*DOWN)
        
        title_sub = Text("Algoritmos Básicos", font_size = 60).to_edge(LEFT, buff = 0.8)
        
        title_info = VGroup(Text("Gabriel Fabian Tortoretto", font_size = 30, fill_opacity=0.5), Text("05/02/2024", font_size = 20, fill_opacity=0.5)).arrange(DOWN).shift(2.5*DOWN)
        
        self.play(FadeIn(lateral_bar), Transform(title, title_sub), FadeIn(trans_title), FadeIn(title_info))
        #self.play(FadeIn(trans_title))
        
        underline_title_sub = Underline(title_sub)
        
        self.play(Create(underline_title_sub), run_time = 1.2)
        
        banner = ManimBanner().scale(0.2).to_corner(DOWN + RIGHT)
        
        self.play(banner.create())
        self.play(banner.expand(direction="left"), run_time = 0.5)
        #self.wait()
        #self.play(Unwrite(banner), run_time = 0.8)
        
        self.wait(2)
        
        pass