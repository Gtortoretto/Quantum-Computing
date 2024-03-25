from manim import *
import numpy as np
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
        
        self.next_section(skip_animations = True)
        
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
        
        title_upper_left.next_to(Line(rectangle_1.get_left(), rectangle_1.get_center()).get_center(), buff=(title_upper_left.get_left() - title_upper_left.get_right())/2).shift(0.7*UP)
        title_upper_right.next_to(Line(rectangle_1.get_right(), rectangle_1.get_center()).get_center(), buff=(title_upper_right.get_left() - title_upper_right.get_right())/2).shift(0.7*UP)
        

        title_separador = VGroup(separador, title_upper_left, title_upper_right)


        # Add the cross separator to the scene
        self.play(FadeIn(title_separador), run_time = 4)
        
        self.wait(3)

        # endregion
        
        self.next_section(skip_animations = True)
        
        # region MyRegion
        

        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{braket}")

        content_bottom_left = (VGroup(Text("0", color = BLACK, font_size = 50), Text("1", color = BLACK, font_size = 50)).arrange(DOWN, buff=0.5))
        #content_bottom_right = (VGroup(Tex(r"\ket{0}", color = BLACK, font_size = 50), Text("1", color = BLACK, font_size = 50)).arrange(DOWN, buff=0.5))
        content_bottom_right = VGroup(Tex(r"$|0\rangle$", color = BLACK, font_size = 75), Tex(r"$|1\rangle$", color = BLACK, font_size = 75)).arrange(DOWN, buff=0.5)
        
        content_bottom_left.next_to(rectangle_1.get_left() + [temp_height/4, 0, 0], buff=-0.5).shift(1.7*DOWN)
        content_bottom_right.next_to(rectangle_1.get_right() - [temp_height/4, 0, 0], buff=-0.5).shift(1.7*DOWN)
        
        separator_text = VGroup(title_upper_left, title_upper_right, content_bottom_left, content_bottom_right)
        
        total_separador = VGroup(separador, separator_text)
        
        
        self.play(FadeIn(content_bottom_left), run_time = 3)
        self.wait(2)
        self.play(FadeIn(content_bottom_right), run_time = 3)
        
        
        
        self.wait(2)
        
        # endregion
        
        self.next_section(skip_animations = True)
        
        # region MyRegion
        
        title_2 = Text("Qubit", font_size = 60).set_color(BLACK).to_edge(UP, buff = 0.8)
        
        m_superposition = MathTex(r"| \psi \rangle =", r"\alpha ", r"|0 \rangle", r" +", r"\beta", r" |1 \rangle", color = BLACK, font_size = 75)
        m_superposition_2 = MathTex(r"\alpha, \beta \in \mathbb{C}", color = BLACK, font_size = 60).next_to(m_superposition, DOWN, buff=1)
        
        vgroup_superposition = VGroup(m_superposition, m_superposition_2)
        
        self.play(FadeOut(total_separador, title), run_time = 3)
        self.play(AnimationGroup(FadeIn(title_2), Write(m_superposition)), run_time = 3)
        self.play(FadeIn(m_superposition_2), run_time = 2)
        
        self.wait(3)
        
        self.play(FadeOut(vgroup_superposition), run_time = 2)
        
        axes = Axes(x_range = (-5, 5), y_range = (-5, 5), x_length = 5, y_length = 5, axis_config = {"color": BLACK, "include_tip": True, "tip_height": (temp_valor := 0.26), "tip_width" : temp_valor})
        plane = NumberPlane(x_range = (-5, 5), y_range = (-5, 5), x_length = 5, y_length = 5, background_line_style={"stroke_opacity": 0.3})
        
        #lable = axes.get_axis_labels(Tex("$\ket{1}$", color = BLACK, font_size = 60, tex_template=myTemplate), Tex("$\ket{0}$", color = BLACK, font_size = 60, tex_template=myTemplate))
        
        x_label = Tex("$\ket{1}$", color = BLACK, font_size = 60, tex_template=myTemplate).scale(0.8)
        y_label = Tex("$\ket{0}$", color = BLACK, font_size = 60, tex_template=myTemplate).scale(0.8)

        x_label.next_to(axes.get_x_axis(), RIGHT, buff=0.1).shift(0.2*UP)
        y_label.next_to(axes.get_y_axis(), UP, buff=0.1).shift(0.3*RIGHT)
        
        lable = VGroup(x_label, y_label)        
        
        grafico = VGroup(plane, axes, lable).shift(0.9*DOWN)
        
        self.play(FadeIn(grafico), run_time = 2)
        
        m_superposition_new = m_superposition.copy().scale(0.8).to_edge(RIGHT, buff = 1)
        
        self.play(AnimationGroup(grafico.animate.shift(2.7*LEFT), FadeIn(m_superposition_new)), run_time = 2)
        
        ponto_coords = (random.uniform(2, 3), random.uniform(2, 3))
        
        ponto = Dot(axes.c2p(*(ponto_coords)), color = "#be4720")
        
        self.play(FadeIn(ponto))
        
        self.play(AnimationGroup(Indicate(m_superposition_new[1], color = "#be4720"), Flash(m_superposition_new[1], color = "#be4720", flash_radius = 0.3)))
        
        self.play(AnimationGroup(Write(alpha_1 := DashedLine(ponto.get_center(), axes.c2p(ponto_coords[0], 0), color = BLACK)), (alpha := m_superposition_new[1].copy()).animate.next_to(axes.c2p(ponto_coords[0], 0), DOWN), lag_ratio=0.1), run_time = 2)        
        
        self.play(AnimationGroup(Indicate(m_superposition_new[4], color = "#be4720"), Flash(m_superposition_new[4], color = "#be4720", flash_radius = 0.3)))
        
        self.play(AnimationGroup(Write(beta_1 := DashedLine(ponto.get_center(), axes.c2p(0, ponto_coords[1]), color = BLACK)), (beta := m_superposition_new[4].copy()).animate.next_to(axes.c2p(0, ponto_coords[1]), LEFT), lag_ratio=0.1), run_time = 2)
        
        self.wait(0.5)
        
        vetor = Arrow(start = axes.c2p(0, 0), end = axes.c2p(*ponto_coords), color = "#be4720", buff=0)
        
        vetor_text = Tex(r"$\ket{\psi}$", color = BLACK, font_size = 60, tex_template=myTemplate).next_to(vetor, RIGHT, buff=0.2)
        
        self.play(AnimationGroup(FadeOut(VGroup(alpha_1, alpha, beta_1, beta)), FadeIn(vetor), Write(vetor_text), lag_ratio=0.2), run_time = 2)
        
        self.wait(5)
        
        vgroup_grafico = VGroup(grafico, ponto, vetor, vetor_text)
        
        # endregion
        
        self.next_section(skip_animations = True)
        
        # region MyRegion
        
        #self.play(AnimationGroup(FadeOut(vgroup_grafico), m_superposition_new.animate.move_to(ORIGIN, aligned_edge=UP), lag_ratio = 0.5), run_time = 2.5)
    
        self.play(AnimationGroup(Indicate(m_superposition_new[2], color = "#be4720"), Flash(m_superposition_new[2], color = "#be4720", flash_radius = 0.3)))    
        self.play(AnimationGroup(Indicate(m_superposition_new[5], color = "#be4720"), Flash(m_superposition_new[5], color = "#be4720", flash_radius = 0.3)))    
            
        
        #self.play(FadeIn(arrow_zero := Arrow(start = (zero := m_superposition_new[2].copy()).get_center(), end = (temp_end := m_superposition_new[2].get_center() + [-1.5, -2, 0]), color = BLACK, buff=0.7)), run_time = 1)   
        
        #new_zero = Tex(r"$|\alpha|^2$", color = BLACK, font_size = 75, tex_template=myTemplate).scale(0.8).next_to(arrow_zero, DOWN, buff=0.2)
        
        #self.play(ReplacementTransform(zero, new_zero), run_time = 1)
        
        #self.play(Transform(zero, Tex(r"$|\alpha|^2$", color = BLACK, font_size = 75, tex_template=myTemplate).scale(0.8).next_to(arrow_zero, DOWN, buff=0.2)), run_time = 1)
        
        self.play(AnimationGroup((FadeIn(arrow_zero := Arrow(start = (zero := m_superposition_new[2].copy()).get_center(), end = m_superposition_new[2].get_center() + [0, -2, 0], color = BLACK, buff=0.7))), Transform(zero, Tex(r"$|\alpha|^2$", color = BLACK, font_size = 75, tex_template=myTemplate).scale(0.8).next_to(arrow_zero, DOWN, buff=0.2)), lag_ratio=0.1), run_time = 2)
        
        self.play(AnimationGroup((FadeIn(arrow_one := Arrow(start = (one := m_superposition_new[5].copy()).get_center(), end = m_superposition_new[5].get_center() + [0, -2, 0], color = BLACK, buff=0.7))), Transform(one, Tex(r"$|\beta|^2$", color = BLACK, font_size = 75, tex_template=myTemplate).scale(0.8).next_to(arrow_one, DOWN, buff=0.2)), lag_ratio=0.1), run_time = 2)
        
        vgroup_zero_one = VGroup(arrow_zero, zero, arrow_one, one)
        
        self.play(AnimationGroup(FadeOut(vgroup_zero_one), Write((prob := MathTex(r"|\alpha|^2 + |\beta|^2 = 1", color = BLACK, font_size = 60, tex_template=myTemplate).scale(0.8).next_to(arrow_zero, DOWN))), lag_ratio = 0.5), run_time = 2.5)
        
        self.play(FadeOut(vetor, vetor_text), run_time = 2)
        
        linha = Line(start = axes.c2p(0, 0), end = axes.c2p(*ponto_coords), color = BLACK, stroke_width = 4, buff = 0.05)
        linha_text = Text(r"1", color = BLACK, font_size = 40).next_to(linha, RIGHT, buff=0.2)
        
        self.play(AnimationGroup(FadeIn(linha), Write(linha_text), lag_ratio = 0.2), run_time = 2)
        
        self.play(FadeOut(linha_text), run_time = 2)
        
        path_circulo = Circle(radius = (raio := np.linalg.norm(np.array(ponto_coords)))).move_to(axes.c2p(*ponto_coords))
        
        
        k = ValueTracker(temp_valor_inicial := math.atan2(ponto_coords[1], ponto_coords[0]))
        
        """
        
        linha_path = always_redraw(lambda : Line(start = axes.c2p(0, 0), end = axes.c2p(*raio*np.array([np.cos(k.get_value()), np.sin(k.get_value())])), color = BLACK, stroke_width = 4, buff = 0.05))
                
        def circle_func(t):
            return axes.c2p(*raio * np.array([np.cos(t), np.sin(t)]))
        
        circunferencia_path = always_redraw(lambda: axes.add(ParametricFunction(circle_func, t_range=[temp_valor_inicial, k.get_value()], color=BLACK)))
           
        ponto_path = always_redraw(lambda : Dot(axes.c2p(*raio*np.array([np.cos(k.get_value()), np.sin(k.get_value())])), color = "#be4720"))
        
        self.remove(linha, ponto)
        
        self.add(linha_path, circunferencia_path, ponto_path)
        
        self.play(k.animate.set_value(temp_valor_inicial + 2*PI), rate_func = linear, run_time = 4)
        
        """
        def circle_func(t):
            return axes.c2p(*raio * np.array([np.cos(t), np.sin(t)]))
        
         
        linha_path = Line(start=axes.c2p(0, 0), end=axes.c2p(*raio*np.array([np.cos(k.get_value()), np.sin(k.get_value())])), color=BLACK, stroke_width=4, buff=0.05)
        
        circunferencia_path = ParametricFunction(circle_func, t_range=[temp_valor_inicial, temp_valor_inicial], color=BLACK)
        
        ponto_path = Dot(axes.c2p(*raio*np.array([np.cos(k.get_value()), np.sin(k.get_value())])), color="#be4720")


        def update_line(mob):
            mob.become(Line(start=axes.c2p(0, 0), end=axes.c2p(*raio*np.array([np.cos(k.get_value()), np.sin(k.get_value())])), color=BLACK, stroke_width=4, buff=0.05))

        def update_circunferencia(mob):
            mob.become(ParametricFunction(circle_func, t_range=[temp_valor_inicial, k.get_value()], color=BLACK))

        def update_dot(mob):
            mob.move_to(axes.c2p(*raio*np.array([np.cos(k.get_value()), np.sin(k.get_value())])))

        linha_path.add_updater(update_line)
        circunferencia_path.add_updater(update_circunferencia)
        ponto_path.add_updater(update_dot)

        self.remove(linha, ponto)

        self.add(linha_path, circunferencia_path, ponto_path)

        self.play(k.animate.set_value(temp_valor_inicial + 2*PI), rate_func=linear, run_time=4)

        linha_path.remove_updater(update_line)
        circunferencia_path.remove_updater(update_circunferencia)
        ponto_path.remove_updater(update_dot)
        
        self.wait(2)
        
        
        # endregion
        
        self.next_section(skip_animations = True)
        
        # region MyRegion
        
        self.wait()
        
        m_superposition = MathTex(r"| \psi \rangle =", r"\alpha ",r"|0 \rangle", r" +", r"\beta", r" |1 \rangle", color = BLACK, font_size = 75)
        
        self.play(AnimationGroup(FadeOut(circunferencia_path, ponto_path, linha_path, grafico, prob), ReplacementTransform(m_superposition_new, m_superposition.scale(1.2).to_edge(UP, buff = 3.4)), lag_ratio=0.7), run_time = 3)
        
        m_superposition_2 = MathTex(r"\alpha, \beta \in ", r"\mathbb{C}", color = BLACK, font_size = 60).next_to(m_superposition, DOWN, buff=1.5)
        
        self.play(FadeIn(m_superposition_2), run_time = 2)
        
        self.play(AnimationGroup(Indicate(m_superposition_2[1], color = "#be4720"), Flash(m_superposition_2[1], color = "#be4720", flash_radius = 0.3)))
        
        self.wait()
        
        m_complexo = MathTex(r"Z = ", r"a + i b", color = BLACK, font_size = 60).next_to(m_superposition, DOWN, buff=1.5)
        
        self.play(AnimationGroup(FadeOut(m_superposition_2), Write(m_complexo), lag_ratio = 1), run_time = 2)
        
        self.wait()
        
        self.play(Transform(m_complexo[1], MathTex(r"re^{i \theta}", color = BLACK, font_size = 60).align_to(m_complexo[1], LEFT).align_to(m_complexo[0], DOWN)), run_time = 2)
        
        m_superposition_new = MathTex(r"| \psi \rangle =", r"r_0 e^{i \theta_0}",r"|0 \rangle", r" +", r"r_1 e^{i \theta_1}", r" |1 \rangle", color = BLACK, font_size = 75)
        
        self.play(TransformMatchingTex(m_superposition, m_superposition_new), run_time = 2)
        
        m_superposition_new_1 = MathTex(r"| \psi \rangle =", r"e^{i \theta_0}",r"( r_0 |0 \rangle", r" +", r"r_1 e^{i (\theta_1 - \theta_0)}", r" |1 \rangle )", color = BLACK, font_size = 75)
        
        m_superposition_new_1.save_state()
        
        self.wait(2)
        
        self.play(TransformMatchingTex(m_superposition_new, m_superposition_new_1), run_time = 2)
        
        self.wait()
        
        self.play(AnimationGroup(Indicate(m_superposition_new_1[1], color = "#be4720"), Flash(m_superposition_new_1[1], color = "#be4720", flash_radius = 0.6)), FadeOut(m_complexo), run_time = 2)
        
        self.wait()
        
        m_superposition_new_2_new = MathTex(r"| \psi ' \rangle =", r"r_0 |0 \rangle", r" +", r"r_1 e^{i (\theta_1 - \theta_0)}", r" |1 \rangle", color = BLACK, font_size = 75).next_to(m_superposition_new_1, DOWN, buff=1.5).shift(0.7*UP)
        
        m_superposition_new_2 = MathTex(r"| \psi ' \rangle =", r"e^{-i \theta_0}| \psi \rangle", color = BLACK, font_size = 75).next_to(m_superposition_new_1, DOWN, buff=1.5)
        
        self.play(AnimationGroup(m_superposition_new_1.animate.shift(0.7*UP), Write(m_superposition_new_2.shift(0.7*UP))), run_time = 2)
        
        self.wait(2)
        
        self.play(TransformMatchingTex(m_superposition_new_2, m_superposition_new_2_new), run_time = 2)

        m_superposition_new_1_scale = m_superposition_new_1.copy().scale(0.60).to_edge(LEFT, buff = 0.63).shift(0.3*DOWN)
        
        m_superposition_new_2_scale = m_superposition_new_2_new.copy().scale(0.60).to_edge(RIGHT, buff = 0.63).align_to(m_superposition_new_1_scale, DOWN)
        
        self.wait()
        
        self.play(AnimationGroup(Transform(m_superposition_new_1, m_superposition_new_1_scale), ReplacementTransform(m_superposition_new_2_new, m_superposition_new_2_scale), lag_ratio = 0), run_time = 2)
        
        self.wait()
        
        expectation_value = MathTex(r"\Braket{\psi '|\mathcal{O}|\psi '}", color = BLACK, font_size = 60, tex_template = myTemplate).shift(1.7*DOWN)
        
        self.play(Write(expectation_value), run_time = 2)
        
        self.wait()
        
        self.play(TransformMatchingTex(expectation_value, (expectation_value := MathTex(r"\Braket{\psi '|\mathcal{O}|\psi '}", r"=", r"\Braket{e^{i \theta_0}\psi |\mathcal{O}|e^{-i \theta_0}\psi }", color = BLACK, font_size = 60, tex_template = myTemplate).shift(1.7*DOWN))), run_time = 2)
        
        self.wait()
        
        self.play(TransformMatchingTex(expectation_value, (expectation_value := MathTex(r"\Braket{\psi '|\mathcal{O}|\psi '}", r"=", r"e^{i \theta_0}e^{-i \theta_0}\Braket{\psi  |\mathcal{O}|\psi }", color = BLACK, font_size = 60, tex_template = myTemplate).shift(1.7*DOWN))), run_time = 2)
        
        self.wait(1)
        
        self.play(TransformMatchingTex(expectation_value, (expectation_value := MathTex(r"\Braket{\psi '|\mathcal{O}|\psi '}", r"=", r"e^{i \theta_0}e^{-i \theta_0}\Braket{\psi  |\mathcal{O}|\psi }", r"= \Braket{\psi|\mathcal{O}|\psi}", color = BLACK, font_size = 60, tex_template = myTemplate).shift(1.7*DOWN))), run_time = 2)
        
        self.wait()
        
        self.play(AnimationGroup(FadeOut(expectation_value), FadeOut(m_superposition_new_2_scale), Restore(m_superposition_new_1), lag_ratio = 0.6), run_time = 3.7)
    
        self.wait()

        self.play(TransformMatchingTex(m_superposition_new_1, (m_superposition_final := MathTex(r"| \psi \rangle =",r" r_0 |0 \rangle", r" +", r"r_1 e^{i (\theta_1 - \theta_0)}", r" |1 \rangle", color = BLACK, font_size = 75))), run_time = 2)
        
        delta = MathTex(r"\delta = \theta_1 - \theta_0", color = BLACK, font_size = 60).next_to(m_superposition_final, DOWN, buff=1.5)
        
        self.wait()
        
        self.play(Write(delta), run_time = 2)
        
        self.wait()
        
        self.play(AnimationGroup(TransformMatchingTex(m_superposition_final, (m_superposition_final := MathTex(r"| \psi \rangle =",r" r_0 |0 \rangle", r" +", r"r_1 e^{i \delta}", r" |1 \rangle", color = BLACK, font_size = 75))), FadeOut(delta), lag_ratio = 0.5), run_time = 2)
        
        self.wait()
        
        m_superposition_final_2 = MathTex(r"|r_0|^2", r"+", r"|r_1 e^{i\delta}|^2", r"=", r"1", color = BLACK, font_size = 75, tex_template=myTemplate).next_to(m_superposition_final, DOWN, buff=1.5)
        
        self.play(Write(m_superposition_final_2), run_time = 2)
        
        self.wait()
        
        self.play(TransformMatchingTex(m_superposition_final_2, (m_superposition_final_2 := MathTex(r"|r_0|^2", r"+", r"|r_1 e^{i\delta}|^2", r"=", r"1", r"\quad ; \quad", r"r_1 e^{i\delta}", r"\in \mathbb{C}", color = BLACK, font_size = 75, tex_template=myTemplate).next_to(m_superposition_final, DOWN, buff=1.5))), run_time = 2)
        
        self.wait()
        
        self.play(TransformMatchingTex(m_superposition_final_2, (m_superposition_final_2 := MathTex(r"|r_0|^2", r"+", r"|r_1 e^{i\delta}|^2", r"=", r"1", r"\quad ; \quad", r"r_1 e^{i\delta}", r" = x + yi", color = BLACK, font_size = 75, tex_template=myTemplate).next_to(m_superposition_final, DOWN, buff=1.5))), run_time = 2)
        
        self.wait()
        
        self.play(FadeOut(m_superposition_final_2), run_time = 2)
        
        m_superposition_final_2 = MathTex(r"|r_0|^2", r"+", r"|r_1 e^{i\delta}|^2", r"=", r"r_0^2",r"+",r"(x + iy)^*(x + iy)", color = BLACK, font_size = 75, tex_template=myTemplate).next_to(m_superposition_final, DOWN, buff=1.5)
        
        self.play(Write(m_superposition_final_2), run_time = 2)
        
        self.wait()
        
        self.play(TransformMatchingTex(m_superposition_final_2, (m_superposition_final_2 := MathTex(r"|r_0|^2", r"+", r"|r_1 e^{i\delta}|^2", r"=", r"r_0^2",r"+",r"(x-iy)(x+iy)", color = BLACK, font_size = 75, tex_template=myTemplate).next_to(m_superposition_final, DOWN, buff=1.5))), run_time = 2)

        self.wait()
        
        self.play(TransformMatchingTex(m_superposition_final_2, (m_superposition_final_2 := MathTex(r"|r_0|^2", r"+", r"|r_1 e^{i\delta}|^2", r"=", r"r_0^2",r"+",r"x^2 + y^2", color = BLACK, font_size = 75, tex_template=myTemplate).next_to(m_superposition_final, DOWN, buff=1.5))), run_time = 2)
        
        self.wait()
        
        self.play(AnimationGroup(FadeOut(m_superposition_final_2), Write(m_superposition_final_2 := MathTex(r"\therefore \quad ", r"r_0^2 ", r"+ x^2 + y^2 = 1", color = BLACK, font_size = 75, tex_template=myTemplate).next_to(m_superposition_final, DOWN, buff=1.5)), lag_ratio = 1), run_time = 4)

        self.wait()
        
        self.play(TransformMatchingTex(m_superposition_final_2, (m_superposition_final_2 := MathTex(r"\therefore \quad ", r"z^2", r"+ x^2 + y^2 = 1", r"\quad ; \quad r_0 \equiv z", color = BLACK, font_size = 75, tex_template=myTemplate).next_to(m_superposition_final, DOWN, buff=1.5))), run_time = 2)

        self.wait(2)
        
        self.play(AnimationGroup(FadeOut(m_superposition_final_2, m_superposition_final), Transform(title_2, (title_2 := Text("Esfera de Bloch", font_size = 60).set_color(BLACK).to_edge(UP, buff = 0.8))), lag_ratio = 1), run_time = 2)
        
        
        self.wait(2) 
        
        
        # endregion
        
        self.next_section(skip_animations = False)
        
        # region MyRegion
        
        titulo_3 = Text("Coordenadas Esféricas :", font_size = 36).set_color(BLACK).next_to(title_2, DOWN, buff = 1.5).to_edge(LEFT, buff = 1)
                
        self.play(Write(titulo_3), run_time = 2)
        
        coordenadas_esfericas = VGroup(
            
            MathTex(r"x = r \sin(\theta) \cos(\phi)", color = BLACK, font_size = 60, tex_template=myTemplate).next_to(titulo_3, DOWN, buff=1.5),
            
            MathTex(r"y = r \sin(\theta) \sin(\phi)", color = BLACK, font_size = 60, tex_template=myTemplate),
            
            MathTex(r"z = r \cos(\theta)", color = BLACK, font_size = 60, tex_template=myTemplate)
            
        ).arrange(DOWN, aligned_edge=LEFT, buff = 1).next_to(titulo_3, RIGHT, aligned_edge=UP, buff=1).shift(0.1*UP)
        
        for a in coordenadas_esfericas:
            self.play(Write(a), run_time = 1.5)
        
        self.wait(2)
        
        self.play(AnimationGroup(FadeOut(titulo_3), coordenadas_esfericas.animate.scale(0.8).to_edge(LEFT, buff = 0.8), lag_ratio = 0.6), run_time = 2)
        
        #self.play
        
        m_superposition_final = MathTex(r"\ket{\psi}",r"=",r"r_0",r"\ket{0}",r"+",r"r_1e^{i\delta}",r"\ket{1}", color = BLACK, tex_template=myTemplate, font_size = 70).move_to(coordenadas_esfericas.get_center()).to_edge(RIGHT, buff = 0.8)
        
        self.play(Write(m_superposition_final), run_time = 2)
        
        self.play(TransformMatchingTex(m_superposition_final, (m_superposition_final := MathTex(r"\ket{\psi}",r"=",r"z",r"\ket{0}",r"+",r"(x + yi)",r"\ket{1}", color = BLACK, tex_template=myTemplate, font_size = 70).move_to(coordenadas_esfericas.get_center()).to_edge(RIGHT, buff = 0.5))), run_time = 2)
        
        self.wait()
        
        self.wait(2)
        
        # endregion
        
        pass
        


class Test(Scene):
        
        def construct(self):
        
            ponto_coords = (random.uniform(2, 3), random.uniform(2, 3))
            
            path_circulo = Circle(radius = (raio := np.linalg.norm(np.array(ponto_coords))))
            
            k = ValueTracker(temp_valor_inicial := math.atan2(ponto_coords[1], ponto_coords[0]))
                    
            def circle_func(t):
                return raio*np.cos(t), raio*np.sin(t), 0
            
            circunferencia_path = always_redraw(lambda: ParametricFunction(circle_func, t_range=[temp_valor_inicial, k.get_value()]))
            
            self.play(FadeIn(circunferencia_path))
            
            self.play(k.animate.set_value(temp_valor_inicial + 2*PI), rate_func = linear, run_time = 4)
            
            self.play(FadeOut(circunferencia_path))
            
            self.wait()
            

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
