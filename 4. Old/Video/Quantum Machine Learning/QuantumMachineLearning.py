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

class PatternClassification(Scene):
    
    def construct(self):
        
        lateral_bar = Rectangle(height = config.frame_height, width = (comprimento_barra_lateral := 1.5), fill_color = "#353537", fill_opacity = 1, stroke_width = 0).to_edge(RIGHT, buff = 0)
        
        self.camera.background_color = "#ffffff"
            
        title = Text("Pattern Classification", font_size = 60).set_color(BLACK).to_edge(UP, buff = 0.8)
        
        self.play(Write(title), run_time = 2)
        self.play(title.animate.to_edge(LEFT, buff = 0.8), FadeIn(lateral_bar))
        self.wait()
        
        bp_1 = Text("Algoritmo de k-vizinhos mais próximos", font_size = 30, fill_color = BLACK).move_to(ORIGIN - [comprimento_barra_lateral, 0, 0])
        
        self.play(Write(bp_1), run_time = 2)
        
        bp_1_altered = bp_1.copy().next_to(title, DOWN, buff = (buff_bp_1 := 0.59))
    
        self.play(bp_1.animate.next_to(title, DOWN, buff = buff_bp_1), Create(Dot(color=BLACK).next_to(bp_1_altered, LEFT, buff = 0.3)))
        
        self.play(Write(Text("(k-nearest neighbor algorithm)", slant = ITALIC, color = BLACK, font_size = 26).next_to(bp_1, DOWN, buff=0.2, aligned_edge=LEFT)), run_time = 2)
        
        self.wait()
        
        bp_2 = Text("Conjunto de treinamento: ", font_size = 30, color = BLACK)
        
        tau = MathTex(r"\tau", color = BLACK).scale(1.2)
        
        bp_2_vgroup = VGroup(bp_2, tau).arrange(RIGHT).move_to(ORIGIN - [comprimento_barra_lateral - 0.3, 0, 0])
        
        self.play(Write(bp_2_vgroup))
        
        grafico = Axes(
            x_range = (0, 9, 1),
            y_range = (0, 10, 1.5),
            axis_config = {"color": BLACK},
            tips = True
        ).scale(0.4).next_to(bp_2_vgroup, DOWN, buff = 0.5)
        
        self.play(Write(grafico), run_time = 1.5)
        
        pontos_do_grafico = [Dot(grafico.c2p(x, y), color = "#6087cf") for x, y in [((random.uniform(0, 8)), random.uniform(0, 9)) for _ in range(20)]]
        
        self.play(LaggedStart(*[Write(ponto) for ponto in pontos_do_grafico], lag_ratio = 0.02))
        
        self.play(Flash(tau, color = "#6087cf", flash_radius = 0.2))
        
        self.play(FadeOut(bp_2_vgroup))
        
        bp_3 = Text("Vetor não classificado: ", font_size = 30, color = BLACK)
        
        x_vetor = MathTex(r"\vec{x}", color = BLACK).scale(1)
        
        bp_3_vgroup = VGroup(bp_3, x_vetor).arrange(RIGHT).move_to(ORIGIN - [comprimento_barra_lateral - 0.3, 0, 0])
        
        ponto_vetor = Dot(grafico.c2p(random.uniform(4, 6), random.uniform(3, 7)), color = "#be4720")
        
        self.play(Write(ponto_vetor))

        self.play(*list(FadeOut(ponto) for ponto in pontos_do_grafico))
        
        #linha = Line(grafico.coords_to_point(0,0), ponto_vetor, color = "#454c5a").set_opacity(0.5)
        
        #brace_linha = Brace(linha, direction = linha.copy().rotate(PI/2).get_unit_vector(), color = BLACK)
        #brace_linha_text = brace_linha.get_tex(r"\vec{x}", buff = 0.1).set_color(BLACK)

        #self.play(Write(bp_3_vgroup), Write(linha))
        #self.play(FadeIn(brace_linha), Write(brace_linha_text))

        vetor_linha = Arrow(grafico.coords_to_point(0,0), ponto_vetor, color = "#454c5a", max_tip_length_to_length_ratio=0.10, buff = 0.1).set_opacity(1)
        
        vetor_linha_text = MathTex(r"\vec{x}", color = BLACK).next_to(vetor_linha.get_center(), UP, buff = 0.35)
        
        
        self.play(Write(bp_3_vgroup))
        self.play(Write(vetor_linha), Write(vetor_linha_text), run_time = 1.5)

        self.wait()

        self.play(FadeOut(bp_3_vgroup), FadeOut(vetor_linha), FadeOut(vetor_linha_text))

        self.play(*[FadeIn(ponto) for ponto in pontos_do_grafico], lag_ratio = 0.02)
        
        bp_4 = Text("Determinar a classe: ", font_size = 30, color = BLACK)
        bp_4_tex = MathTex(r"c^{x}", color = BLACK).scale(1)
        bp_4_vgroup = VGroup(bp_4, bp_4_tex).arrange(RIGHT).move_to(ORIGIN - [comprimento_barra_lateral - 0.3, 0, 0])
        
        self.play(Write(bp_4_vgroup))
    
        linhas_ponto_grafico = [DashedLine(ponto_vetor.get_center(), ponto.get_center(), color = "#454c5a").set_opacity(0.5) for ponto in pontos_do_grafico]
        
        self.play(*[Write(linha) for linha in linhas_ponto_grafico], run_time = 2)
        
        self.play(*[FadeOut(linha) for linha in linhas_ponto_grafico], run_time = 1.5)
        
        bp_4_vetor = Arrow([0, 0, 0], [1.8, 0, 0], color = "#454c5a", max_tip_length_to_length_ratio=0.15, buff = 0).set_opacity(1)
        
        self.play(*[FadeOut(ponto) for ponto in pontos_do_grafico], lag_ratio = 0.02)
        
        self.play(Write((temp_group := VGroup(bp_4_vetor, MathTex(r"\{\vec{x}, c^{x}\}", color = BLACK).scale(1)).arrange(RIGHT).next_to(ponto_vetor, RIGHT, buff = 0.1))), run_time = 1.5)
    
        self.play(FadeOut(bp_4_vgroup), FadeOut(temp_group))
        
        self.play(*[FadeIn(ponto) for ponto in pontos_do_grafico], lag_ratio = 0.02)
        
        group_final = VGroup(grafico, ponto_vetor, *[ponto for ponto in pontos_do_grafico])
        
        self.play(group_final.animate.shift(1.5*LEFT + 0.5*UP), run_time = 1.5)
        
        bp_5 = MathTex(r"\tau = \{ \vec{v}_p , c^{p} \}_{p = 1, 2, \dots, N}", color = BLACK).scale(1.2)
        
        self.play(Write(bp_5.next_to(group_final, buff = 1.72)), run_time = 2)
        
        self.wait(2)
        
        self.play(*[FadeOut(objects) for objects in self.mobjects])
        
class KNN(Scene):
    
    def construct(self):
    
        self.section1()
        #self.section2()
    
    def section1(self):
         
        self.next_section(skip_animations = False)
        
        # region MyRegion
        
        lateral_bar = Rectangle(height = config.frame_height, width = (comprimento_barra_lateral := 1.5), fill_color = "#353537", fill_opacity = 1, stroke_width = 0).to_edge(RIGHT, buff = 0)
        
        self.camera.background_color = "#ffffff"
            
        title = Text("K-Nearest Neighbors", font_size = 60).set_color(BLACK).to_edge(UP, buff = 0.8)
        
        self.play(Write(title), run_time = 2)
        self.play(title.animate.to_edge(LEFT, buff = 0.8), FadeIn(lateral_bar))
        self.wait()
        
        grafico = Axes(
            x_range = (0, 10, 1),
            y_range = (0, 10, 1),
            x_length= 11,
            y_length= 7,
            axis_config = {"color": BLACK},
            tips = True
        ).scale(0.6).move_to(ORIGIN - [comprimento_barra_lateral - 0.5, 0.7, 0])
        
        grafico.get_x_axis().set_color(BLACK), grafico.get_y_axis().set_color(BLACK)
        
        self.play(Write(grafico), run_time = 1.5)
        
        #pontos_grupo1 = [Dot(grafico.c2p(x, y), color = "#fbb84a") for x, y in [((random.uniform(0, 8)), random.uniform(0, 9)) for _ in range(20)]]
        
        #pontos_grupo2 = [Dot(grafico.c2p(x, y), color = "#a18c8a") for x, y in [((random.uniform(0, 8)), random.uniform(0, 9)) for _ in range(20)]]
        
        #pontos_grupo3 = [Dot(grafico.c2p(x, y), color = "#6087cf") for x, y in [((random.uniform(0, 8)), random.uniform(0, 9)) for _ in range(20)]]

        def generate_points_in_circle(x0, y0, r, num_points):
            
            return [(x0 + random.uniform(0, r) * math.cos(random.uniform(0, 2 * math.pi)), y0 + random.uniform(0, r) * math.sin(random.uniform(0, 2 * math.pi))) for _ in range(num_points)]

        centers = {"#fbb84a" : [2, 7], "#a18c8a" : [5, 2], "#6087cf" : [8, 7]}
    
        pontos_grupo1 = [Dot(grafico.c2p(x, y), color = "#fbb84a") for x, y in generate_points_in_circle(*list(centers.values())[0], 2, 20)]

        pontos_grupo2 = [Dot(grafico.c2p(x, y), color = "#a18c8a") for x, y in generate_points_in_circle(*list(centers.values())[1], 2, 20)]

        pontos_grupo3 = [Dot(grafico.c2p(x, y), color = "#6087cf") for x, y in generate_points_in_circle(*list(centers.values())[2], 2, 20)]

        self.play(LaggedStart(*[Write(ponto) for ponto in pontos_grupo1], lag_ratio = 0.02))
        
        self.play(LaggedStart(*[Write(ponto) for ponto in pontos_grupo2], lag_ratio = 0.02))
        
        self.play(LaggedStart(*[Write(ponto) for ponto in pontos_grupo3], lag_ratio = 0.02))
        
        self.play(grafico.animate.set_opacity(0.1), run_time=0.6)

        
        grupo1_label_line = VGroup((temp_line1 := Line(0, 1, color = BLACK)).scale(0.6), (temp_line2 := Line([0, 0, 0], [1, 0, 0], color = BLACK).shift(temp_line1.get_end()))).shift(grafico.c2p(*list(centers.values())[0])).set_opacity(0.5)
        grupo1_label = Text("Grupo 1", font_size = 20, color = BLACK).align_to(temp_line2, DOWN).align_to(temp_line2, RIGHT)
        
        grupo2_label_line = VGroup((temp_line1 := Line(0, 1, color = BLACK)).scale(0.6), (temp_line2 := Line([0, 0, 0], [1, 0, 0], color = BLACK).shift(temp_line1.get_end()))).shift(grafico.c2p(*list(centers.values())[1])).set_opacity(0.5)
        grupo2_label = Text("Grupo 2", font_size = 20, color = BLACK).align_to(temp_line2, DOWN).align_to(temp_line2, RIGHT)
        
        grupo3_label_line = VGroup((temp_line1 := Line(0, 1, color = BLACK)).scale(0.6), (temp_line2 := Line([0, 0, 0], [1, 0, 0], color = BLACK).shift(temp_line1.get_end()))).shift(grafico.c2p(*list(centers.values())[2])).set_opacity(0.5)
        grupo3_label = Text("Grupo 3", font_size = 20, color = BLACK).align_to(temp_line2, DOWN).align_to(temp_line2, RIGHT)
        
        
        self.play(Write(grupo1_label_line), Write(grupo1_label))
        self.play(Write(grupo2_label_line), Write(grupo2_label))
        self.play(Write(grupo3_label_line), Write(grupo3_label))
        
        self.wait(0.5)
        
        grupo1_label_final = MathTex(r"c^1", color = BLACK).scale(1).next_to(grafico.coords_to_point(*list(centers.values())[0]), UP, buff = 1)
        
        grupo2_label_final = MathTex(r"c^2", color = BLACK).scale(1).next_to(grafico.coords_to_point(*list(centers.values())[1]), UP, buff = 1)
        
        grupo3_label_final = MathTex(r"c^3", color = BLACK).scale(1).next_to(grafico.coords_to_point(*list(centers.values())[2]), UP, buff = 1)
        
        self.play(FadeOut(grupo1_label_line), FadeOut(grupo1_label), FadeOut(grupo2_label_line), FadeOut(grupo2_label), FadeOut(grupo3_label_line), FadeOut(grupo3_label), FadeIn(grupo1_label_final), FadeIn(grupo2_label_final), FadeIn(grupo3_label_final))
        
        self.wait(0.7)
        
        self.play(FadeOut(grupo1_label_final), FadeOut(grupo2_label_final), FadeOut(grupo3_label_final))
        
        ponto_n_identificado = Dot(grafico.c2p(4.5, 5.2), color = "#f15b2b")
        
        self.play(Write(ponto_n_identificado))
        
        self.play(*list(ponto.animate.set_opacity(0.1) for ponto in (pontos_grupo1 + pontos_grupo2 + pontos_grupo3)),run_time=0.6)
        
        ponto_n_identificado_label_line = VGroup((temp_line1 := Line(0, 1, color = BLACK)).scale(0.6), (temp_line2 := Line([0, 0, 0], [3, 0, 0], color = BLACK).shift(temp_line1.get_end()))).shift(ponto_n_identificado.get_center()).set_opacity(0.5)
        
        ponto_n_identificado_label = Text("Vetor não identificado", font_size = 20, color = BLACK).align_to(temp_line2, DOWN).align_to(temp_line2, RIGHT).shift(0.1*UP)
        
        self.play(Write(ponto_n_identificado_label_line), Write(ponto_n_identificado_label), run_time = 1.5)
        
        self.wait(0.6)
        
        ponto_n_identificado_label_line_modificado = VGroup((temp_line1 := Line(0, 1, color = BLACK)).scale(0.6), (temp_line2 := Line([0, 0, 0], [0.5, 0, 0], color = BLACK).shift(temp_line1.get_end()))).shift(ponto_n_identificado.get_center()).set_opacity(0.5)
        
        ponto_n_identificado_label_modificado = MathTex(r"\vec{x}", color = BLACK).align_to(temp_line2, DOWN).align_to(temp_line2, RIGHT).shift(0.1*UP)
        
        self.play(ReplacementTransform(ponto_n_identificado_label_line, ponto_n_identificado_label_line_modificado), ReplacementTransform(ponto_n_identificado_label, ponto_n_identificado_label_modificado))
        
        self.play(FadeOut(ponto_n_identificado_label_line_modificado), FadeOut(ponto_n_identificado_label_modificado))
        
        self.play(*list(ponto.animate.set_opacity(1) for ponto in (pontos_grupo1 + pontos_grupo2 + pontos_grupo3)),run_time=0.6)
        
        self.play(*list(FadeOut(ponto) for ponto in (pontos_grupo1 + pontos_grupo2 + pontos_grupo3)), run_time = 1.5)
        
        def generate_points_in_circle_min_distance(center_x, center_y, radius, num_points, min_distance):
            points = []
            for _ in range(num_points):
                while True:
                    x = random.uniform(center_x - radius, center_x + radius)
                    y = random.uniform(center_y - radius, center_y + radius)
                    new_point = (x, y)
                    if all(distance(new_point, p) >= min_distance for p in points):
                        points.append(new_point)
                        break
            return points

        def distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        
        #pontos_grupo1_alternativo = [Dot(grafico.c2p(x, y), color = "#fbb84a") for x, y in generate_points_in_circle(*list(centers.values())[0], 1.5, 4)]
        
        #pontos_grupo2_alternativo = [Dot(grafico.c2p(x, y), color = "#a18c8a") for x, y in generate_points_in_circle(*list(centers.values())[1], 2, 5)]
        
        #pontos_grupo3_alternativo = [Dot(grafico.c2p(x, y), color = "#6087cf") for x, y in generate_points_in_circle(*list(centers.values())[2], 1.2, 3)]

        pontos_grupo1_alternativo = [Dot(grafico.c2p(x, y), color="#fbb84a") for x, y in generate_points_in_circle_min_distance(*list(centers.values())[0], 1.5, 4, min_distance = 1)]
        
        pontos_grupo2_alternativo = [Dot(grafico.c2p(x, y), color="#a18c8a") for x, y in generate_points_in_circle_min_distance(*list(centers.values())[1], 2, 5, min_distance = 1)]
        
        pontos_grupo3_alternativo = [Dot(grafico.c2p(x, y), color="#6087cf") for x, y in generate_points_in_circle_min_distance(*list(centers.values())[2], 1.2, 3, min_distance = 1)]
        
        self.play(LaggedStart(*[Write(ponto) for ponto in pontos_grupo1_alternativo], lag_ratio = 0.02))
        
        self.play(LaggedStart(*[Write(ponto) for ponto in pontos_grupo2_alternativo], lag_ratio = 0.02))
        
        self.play(LaggedStart(*[Write(ponto) for ponto in pontos_grupo3_alternativo], lag_ratio = 0.02))
        
        linha_pontos_grupo1_alternativo = [DashedLine(ponto_n_identificado.get_center(), ponto.get_center(), color = "#454c5a").set_opacity(0.3) for ponto in pontos_grupo1_alternativo]
        
        linha_pontos_grupo2_alternativo = [DashedLine(ponto_n_identificado.get_center(), ponto.get_center(), color = "#454c5a").set_opacity(0.3) for ponto in pontos_grupo2_alternativo]
        
        linha_pontos_grupo3_alternativo = [DashedLine(ponto_n_identificado.get_center(), ponto.get_center(), color = "#454c5a").set_opacity(0.3) for ponto in pontos_grupo3_alternativo]
        
        label_linha_pontos_grupo1_alternativo = [MarkupText(f"d<sub>{n}</sub>", color = BLACK).scale(0.5) for n in range(1, 5)]
        
        label_linha_pontos_grupo2_alternativo = [MarkupText(f"d<sub>{n}</sub>", color = BLACK).scale(0.5) for n in range(5, 10)]
        
        label_linha_pontos_grupo3_alternativo = [MarkupText(f"d<sub>{n}</sub>", color = BLACK).scale(0.5) for n in range(10, 13)]
        
        for a,b in zip(label_linha_pontos_grupo1_alternativo + label_linha_pontos_grupo2_alternativo + label_linha_pontos_grupo3_alternativo, pontos_grupo1_alternativo + pontos_grupo2_alternativo + pontos_grupo3_alternativo):
            a.next_to(b, UP, buff = 0.1)
        
        def check_and_move_overlap(mobjects):
            for i in range(len(mobjects)):
                for j in range(i+1, len(mobjects)):
                    bbox_i = mobjects[i].get_bounding_box()
                    bbox_j = mobjects[j].get_bounding_box()

                    if bbox_i.get_right() > bbox_j.get_left() and bbox_i.get_top() > bbox_j.get_bottom():
                        # i and j are overlapping, move j up and left
                        mobjects[j].shift(UP*0.5)

        def check_and_move_overlap2(mobjects):
            for i in range(len(mobjects)):
                for j in range(i+1, len(mobjects)):
                    bbox_i = mobjects[i].get_center() + np.array([0, mobjects[i].get_height()/2, 0])
                    bbox_j = mobjects[j].get_center() - np.array([0, mobjects[j].get_height()/2, 0])

                    if bbox_i[0] > bbox_j[0] and bbox_i[1] > bbox_j[1]:
                        # i and j are overlapping, move j up and left
                        mobjects[j].shift(UP*0.2)

        #check_and_move_overlap2(label_linha_pontos_grupo1_alternativo + label_linha_pontos_grupo2_alternativo + label_linha_pontos_grupo3_alternativo)
        
        self.play(*[Write(linha) for linha in linha_pontos_grupo1_alternativo], *[Write(label) for label in label_linha_pontos_grupo1_alternativo], run_time = 1.5)
        
        self.play(*[Write(linha) for linha in linha_pontos_grupo2_alternativo], *[Write(label) for label in label_linha_pontos_grupo2_alternativo], run_time = 1.5)
        
        self.play(*[Write(linha) for linha in linha_pontos_grupo3_alternativo], *[Write(label) for label in label_linha_pontos_grupo3_alternativo], run_time = 1.5)
        
        grupo_geral = VGroup(grafico, ponto_n_identificado, *pontos_grupo1_alternativo, *pontos_grupo2_alternativo, *pontos_grupo3_alternativo, *linha_pontos_grupo1_alternativo, *linha_pontos_grupo2_alternativo, *linha_pontos_grupo3_alternativo, *label_linha_pontos_grupo1_alternativo, *label_linha_pontos_grupo2_alternativo, *label_linha_pontos_grupo3_alternativo)
        
        self.play(grupo_geral.animate.shift(2.29*LEFT), run_time = 1.5)
        
        grupo_intermediario = sorted(list(zip(linha_pontos_grupo1_alternativo + linha_pontos_grupo2_alternativo + linha_pontos_grupo3_alternativo, pontos_grupo1_alternativo + pontos_grupo2_alternativo + pontos_grupo3_alternativo, label_linha_pontos_grupo1_alternativo + label_linha_pontos_grupo2_alternativo + label_linha_pontos_grupo3_alternativo)), key = lambda x: x[0].get_length())
        
        '''
        for index, a in enumerate(grupo_intermediario):
            
            self.play(a[0].copy().animate.rotate(angle=-a[0].get_angle(), about_point=a[0].get_start()).shift(RIGHT*1.5 + 1.5*UP), a[1].copy().animate.shift(RIGHT*1), a[2].copy().animate.shift(RIGHT*1), run_time = 1.5)
        
        '''
    
        vgrupo_intermediario = VGroup(*[VGroup(a[2].copy(), a[1].copy(), a[0].copy(), ponto_n_identificado.copy()) for a in grupo_intermediario])
        
        vgrupo_total = VGroup(*[VGroup(a[2].copy(), a[1].copy(), a[0].copy().rotate(angle=-a[0].get_angle(), about_point=a[0].get_start()), ponto_n_identificado.copy()).arrange(LEFT, buff = 0.1) for a in grupo_intermediario]).arrange(DOWN, buff = 0.15, aligned_edge=LEFT).shift(3.1*RIGHT + 0.75*DOWN)
        
        '''
        self.play(*[Transform(temp_1, temp_2) for temp_1, temp_2 in zip(vgrupo_intermediario, vgrupo_total)], run_time = 5, rate_func=linear)
    
        
        for i, (temp_1, temp_2) in enumerate(zip(vgrupo_intermediario, vgrupo_total)):
            # Decrease run_time as we progress through the list
            run_time = 0.7 if i < len(vgrupo_intermediario) / 2 else 0.3
            self.play(Transform(temp_1, temp_2), run_time=run_time)
        
        '''
        
        self.play(AnimationGroup(*[ReplacementTransform(temp_1, temp_2) for temp_1, temp_2 in zip(vgrupo_intermediario, vgrupo_total)], lag_ratio=0.1), run_time = 5, rate_func=linear)
        
        arrow_group = Arrow(vgrupo_total.get_top(), vgrupo_total.get_bottom(), color = '#cc5223').set_opacity(0.7).next_to(vgrupo_total, LEFT, buff = 0.35)
        
        self.play(Write(arrow_group), rate_func=rate_functions.ease_out_sine, run_time = 1.5)

        vgrupo_antigo = VGroup(*[VGroup(a[2], a[1], a[0], ponto_n_identificado) for a in grupo_intermediario], grafico, arrow_group)
        
        self.play(AnimationGroup(FadeOut(vgrupo_antigo), vgrupo_total.animate.to_edge(LEFT, buff = 2.15), lag_ratio = 0.5), rate_func=rate_functions.ease_out_sine, run_time = 2)
        
        # endregion
        
        self.next_section(skip_animations = False)
        
        # region MyRegion
        
        counter_square = RoundedRectangle(corner_radius=0.15, color = "#6d6d6d", stroke_width=1).set_opacity(0.35)
        
        counter_text = Text("Contador: ", font = "Century Gothic", font_size = 30)
        
        counter_square_text = RoundedRectangle(corner_radius=0.15, color = "#5e5e5e", stroke_width=1).set_opacity(0.6).stretch_to_fit_width(counter_text.width + 0.5).stretch_to_fit_height(counter_text.height + 0.5)
        
        counter_text.move_to(counter_square_text)
        counter_square.stretch_to_fit_width(counter_square_text.width + 1.5).stretch_to_fit_height(counter_square_text.height + 2.5).next_to(vgrupo_total, RIGHT, buff = 0.5).shift(0.5*UP)
        
        counter_vgroup_contador = VGroup(counter_square_text, counter_text).move_to(counter_square.get_top() + 0.5*DOWN)
        
        
        counter_circles = VGroup(Circle(radius = 0.3, color = BLACK, stroke_width = 2, fill_color = "#fbb84a", fill_opacity=1.0), Circle(radius = 0.3, color = BLACK, stroke_width = 2, fill_color = "#a18c8a", fill_opacity=1.0), Circle(radius = 0.3, color = BLACK, stroke_width = 2, fill_color = "#6087cf", fill_opacity=1.0)).arrange(RIGHT, buff = 0.5).next_to(counter_square_text, DOWN, buff = 0.5)
        
        counter_circles_text = VGroup(Text("0", color = BLACK, font = "Century Gothic", font_size = 30), Text("0", color = BLACK, font = "Century Gothic", font_size = 30), Text("0", color = BLACK, font = "Century Gothic", font_size = 30)).arrange(RIGHT, buff = 0.9).next_to(counter_circles, DOWN, buff = 0.5)
        
        counter_vgroup_geral = VGroup(counter_square, counter_vgroup_contador, counter_circles, counter_circles_text).shift(0.7*RIGHT)
        
        self.play(AnimationGroup(Write(counter_square), Write(counter_vgroup_contador),lag_ratio = 0.71), run_time = 1.5)
        self.play(Write(counter_circles), run_time = 1.5)
        self.play(Write(counter_circles_text), run_time = 1.5)
        
        counter_line = VGroup(Line([0, 0, 0], [4, 0, 0], color = BLACK), (temp_linha_quadrado := VGroup(Rectangle(width=1.2, height=0.4, color = BLACK).set_fill(BLACK, opacity=1.0), Circle(radius = (temp_radius := 0.2), color = BLACK).set_fill(BLACK, opacity=1.0)).arrange(RIGHT, buff = -temp_radius))).arrange(LEFT, buff = 0).next_to(vgrupo_total.get_left(), RIGHT, buff = -1.72).set_color("#8b8a8c")
        
        counter_line_contador = Text("k = 0", font = "Century Gothic", font_size = 25).next_to(temp_linha_quadrado.get_left(), RIGHT, buff = 0.35)
                
        counter_line_vgroup = VGroup(counter_line, counter_line_contador).align_to(vgrupo_total.get_top(), UP).shift(0.3*UP)
        
        
        self.play(FadeIn(counter_line_vgroup))
        
        def counter_k(k):
          
            vgrupo_total_semlinha = [VGroup(*[sublist[0], sublist[1], sublist[3]]) for sublist in vgrupo_total]
    
            counter_circles_text_alt = VGroup(Text(f"{(vgrupo_total_pontos := [rgb_to_hex(a[1].get_color()) for a in vgrupo_total[:k]]).count('#FBB84A')}", color = BLACK, font = "Century Gothic", font_size = 30), Text(f"{vgrupo_total_pontos.count('#A18C8A')}", color = BLACK, font = "Century Gothic", font_size = 30), Text(f"{vgrupo_total_pontos.count('#6087CF')}", color = BLACK, font = "Century Gothic", font_size = 30)).arrange(RIGHT, buff = 0.9).next_to(counter_circles, DOWN, buff = 0.5)
            
            counter_line_vgroup_copy = counter_line_vgroup.copy()
            counter_line_vgroup_copy.align_to(vgrupo_total[k], UP).shift(0.25*UP)

            displacement = counter_line_vgroup_copy.get_center() - counter_line_vgroup.get_center()

            new_text = Text(f"k = {k}", font = "Century Gothic", font_size = 25).move_to(counter_line_contador.get_center() + displacement)

            animations = AnimationGroup(
                counter_line_vgroup.animate.align_to(vgrupo_total[k], UP).shift(0.25*UP),
                Transform(counter_line_contador, new_text),
                AnimationGroup(*[a.animate.set_opacity(1) for a in vgrupo_total_semlinha], lag_ratio = 0.1),
                lag_ratio=0.015,
            )

            self.play(animations, run_time=2)

            self.play(AnimationGroup(Transform(counter_circles_text, counter_circles_text_alt), AnimationGroup(*[b.animate.set_opacity(0.12) for b in vgrupo_total_semlinha[k:]], lag_ratio = 0.1)), run_time = 1.5)

            self.wait()

            pass
        
        counter_k(5)
        counter_k(2)
        counter_k(8)
            
        self.wait(2)

        self.clear()
        
        self.wait()
        
        # endregion
        
        self.next_section()

class KNN_2(Scene):
    
    def construct(self):

        self.next_section(skip_animations = False)
        
        # region MyRegion
        
        lateral_bar = Rectangle(height = config.frame_height, width = (comprimento_barra_lateral := 1.5), fill_color = "#353537", fill_opacity = 1, stroke_width = 0).to_edge(RIGHT, buff = 0)
        
        self.camera.background_color = "#ffffff"
            
        title = Text("Quantum K-Nearest Neighbors", font_size = 60).set_color(BLACK).to_edge(UP, buff = 0.8)
        
        self.play(Write(title), run_time = 2)
        self.play(title.animate.to_edge(LEFT, buff = 0.5), FadeIn(lateral_bar))
        self.wait()
        

        def wrap_text(text, max_width):
            words = text.split(' ')
            lines = []
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) > max_width:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)
            lines.append(' '.join(current_line))
            return VGroup(*[Text(line, font_size = 30, color = BLACK) for line in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        text_1 = wrap_text("O algoritmo KNN depende da determinação das distancias como forma de classificar um vetor desconhecido", 60).next_to(title, DOWN, buff = 1).to_edge(LEFT, buff = 0.5)
        
        self.play(Write(text_1), run_time = 4)
        
        
        midpoint = (text_1.get_bottom() + DOWN * config.frame_height / 2) / 2
        
        plano = NumberPlane(x_range = (-3, 3), y_range = (-3, 3), x_length = 3, y_length = 3, axis_config = {"color": BLACK}, background_line_style = {"stroke_opacity": 0.3}, tips = False).move_to(midpoint)
         
        self.play(Write(plano), run_time = 1.5)
        
        dots = VGroup(Dot(plano.c2p(-2, -2), color = "#fbb84a"), Dot(plano.c2p(2, 2), color = "#6087cf"))

        def generate_points_in_circle_min_distance(center_x, center_y, radius, num_points, min_distance):
            points = []
            for _ in range(num_points):
                while True:
                    x = random.uniform(center_x - radius, center_x + radius)
                    y = random.uniform(center_y - radius, center_y + radius)
                    new_point = (x, y)
                    if all(distance(new_point, p) >= min_distance for p in points):
                        points.append(new_point)
                        break
            return points

        def distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        dots_2 = VGroup(Dot(plano.c2p(generate_points_in_circle_min_distance(-2, -2, 0.5, 1, 1)), color = "#fbb84a"), Dot(plano.c2p(generate_points_in_circle_min_distance(2, 2, 0.5, 1, 1)), color = "#6087cf"))

        self.play(Write(dots), run_time = 1.5) 
        self.play(Transform(dots, dots_2), run_time = 1.5)
        
        
        self.wait(2)

        # endregion
        
        pass


class teste(Scene):
    
    def construct(self):
        
        initial = VGroup(Square(), Circle())
        final = VGroup(Triangle(), Triangle())

        # Animate the transformation from initial to final
        self.play(Transform(initial, final))
        self.wait(1)
