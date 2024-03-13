from manim import *
import random
import math

class Intro(Scene):
    
    def construct(self):
    
        self.camera.background_color = "#353537"
    
        lateral_bar = Rectangle(height = config.frame_height, width = 0.5, fill_color = "#6f6f74", fill_opacity = 1, stroke_width = 0).to_edge(LEFT, buff = 0)
    
        title = Text("Quantum Computing", font_size = 62)
        
        self.play(Write(title), run_time = 3)
        
        self.wait(1)

        trans_title = VGroup(Text("Computação Quântica", font_size = 40)).arrange(DOWN, aligned_edge=LEFT)
        trans_title.to_corner(UP + LEFT, buff = 0.8).shift(0.5*DOWN)
        
        title_sub = Text("Introdução ", font_size = 60).to_edge(LEFT, buff = 0.8)
        
        title_info = VGroup(Text("Gabriel Fabian Tortoretto", font_size = 30, fill_opacity=0.5), Text("13/03/2024", font_size = 20, fill_opacity=0.5)).arrange(DOWN).shift(2.5*DOWN)
        
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


class Scene2(Scene):
    
    def construct(self):
        
        self.next_section(skip_animations = False)
        
        # region MyRegion
        
        
        #lateral_bar = Rectangle(height = config.frame_height, width = (comprimento_barra_lateral := 1.5), fill_color = "#353537", fill_opacity = 1, stroke_width = 0).to_edge(RIGHT, buff = 0)
        
        grey = "#353537"
        
        self.camera.background_color = "#ffffff"
            
        title = Text("bit e Quantum bit", font_size = 60).set_color(BLACK)
        
        self.play(Write(title), run_time = 3)
        
        self.wait(2) 
        
        self.play(title.animate.to_edge(UP, buff = 0.8), run_time = 2)
        
        self.wait(2) 
                
        separador = VGroup((rectangle_1 := Rectangle(width = (temp_height := 9), height = (temp_width := 0.07), color=grey, fill_color=grey, fill_opacity=1).round_corners(temp_width/2)), (rectangle_2 := Rectangle(width=temp_width, height =( temp_height2 := 5), color=grey, fill_color=grey, fill_opacity=1).round_corners(temp_width/2).shift(1*DOWN)))

        title_upper_left = Text("Clássico", color = BLACK, font_size = 40)
        title_upper_right = Text("Quântico", color = BLACK, font_size = 40)
        content_bottom_left = Text("--", color = BLACK, font_size = 24)
        content_bottom_right = Text("--", color = BLACK, font_size = 24)
              
        
        title_upper_left.next_to(rectangle_1.get_left() + [temp_height/4, 0, 0], buff=-0.5).shift(0.7*UP)
        title_upper_right.next_to(rectangle_1.get_right() - [temp_height/4, 0, 0], buff=-0.5).shift(0.7*UP)
        content_bottom_left.next_to(rectangle_1.get_left() + [temp_height/4, 0, 0], buff=-0.5).shift(1.7*DOWN)
        content_bottom_right.next_to(rectangle_1.get_right() - [temp_height/4, 0, 0], buff=-0.5).shift(1.7*DOWN)
        
        title_upper_left.next_to(Line(rectangle_1.get_left(), rectangle_1.get_center()).get_center(), buff=(title_upper_left.get_left() - title_upper_left.get_right())/2).shift(0.7*UP)
        title_upper_right.next_to(Line(rectangle_1.get_right(), rectangle_1.get_center()).get_center(), buff=(title_upper_right.get_left() - title_upper_right.get_right())/2).shift(0.7*UP)
        
        
        separator_text = VGroup(title_upper_left, title_upper_right, content_bottom_left, content_bottom_right)

        title_separador = VGroup(separador, title_upper_left, title_upper_right)

        total_separador = VGroup(separador, separator_text)

        # Add the cross separator to the scene
        self.play(FadeIn(title_separador), run_time = 4)
        
        self.wait(3)

        # endregion
        
        self.next_section(skip_animations = False)

        pass
        


class Basico(ThreeDScene):
    
    def construct(self):
        
        
        
        
        self.set_camera_orientation(phi=75*DEGREES, theta=0)

        axis = ThreeDAxes().scale(0.5)
        sphere = Sphere().set_fill(BLACK)
        arrow = Arrow().put_start_and_end_on(ORIGIN, UP).set_fill(RED).set_stroke(RED)
        arrow.rotate(axis=RIGHT, angle=PI/2, about_point=ORIGIN)

        ket_0 = Text(r"$\ket{0}$").next_to(axis, np.array([0,0,1]))
        ket_1 = Text(r"$\ket{1}$").next_to(axis, -np.array([0,0,1]))
        ket_p = Text(r"$\ket{+}$").next_to(axis, RIGHT)
        ket_m = Text(r"$\ket{-}$").next_to(axis, LEFT)
        ket_pi = Text(r"$\ket{+i}$").next_to(axis, UP)
        ket_mi = Text(r"$\ket{-i}$").next_to(axis, DOWN)
        axis_name_group = VGroup(ket_0, ket_1, ket_p, ket_m, ket_pi, ket_mi)
        
        qubit = Text(r"$\alpha \ket{0} + \beta \ket{1}$")
        
        gate_list = [Text(r'H').to_corner(UP),
                  Text(r'$R_z(\frac{\pi}{2})$').to_corner(UP),
                  Text(r'$R_x(\frac{\pi}{2})$').to_corner(UP),
                  Text(r'$R_y(\frac{\pi}{2})$').to_corner(UP)]
        
        rotation_list = [(PI, [1,0,1]),
                 (PI/2, [0,0,1]),
                 (PI/2, [1,0,0]),
                 (PI/2, [0,1,0])]
                 
        
        #self.set_camera_orientation(phi=75*DEGREES, theta=0)
        #self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.wait()
        self.play(FadeIn(axis))
        
        self.begin_ambient_camera_rotation(rate = 0.05)
        
        self.add_fixed_orientation_mobjects(ket_0, ket_1, ket_p, ket_m, ket_pi, ket_mi)
        self.play(FadeIn(axis_name_group),
              FadeIn(sphere),
              FadeIn(arrow))
        
        i = 0
        i_last = len(gate_list)
        self.add_fixed_in_frame_mobjects(gate_list[0])
        self.play(Write(gate_list[0]),
              Rotate(arrow, 
                 about_point=ORIGIN,
                 angle=rotation_list[i][0], 
                 axis=np.array(rotation_list[i][1])))
        
        i += 1
        while i < i_last:
            self.add_fixed_in_frame_mobjects(gate_list[i])
            self.play(FadeOut(gate_list[i-1]),
                      FadeIn(gate_list[i]),
                      Rotate(arrow, 
                             about_point=ORIGIN,
                             angle=rotation_list[i][0], 
                             axis=np.array(rotation_list[i][1])))
            i += 1
        
        self.play(Uncreate(gate_list[-1]),
                  Uncreate(axis_name_group),
                  Uncreate(axis),
                  Uncreate(sphere),
                  Uncreate(arrow))
        self.stop_ambient_camera_rotation()
        self.wait()
    
        pass
