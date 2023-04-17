from manimlib.imports import *
from manimlib import *
import random

class NeuralNetwork(Scene):
    def construct(self):

        # Create the input image
        input_image = ImageMobject("axial.png")

        # Create the Neural Network
        myNetwork = NeuralNetworkMobject([9, 4, 4, 1])
        myNetwork.label_inputs("x")
        myNetwork.label_hidden_layers("h")
        myNetwork.label_outputs("\hat{y}")

        # Create the layer label
        output_label = TextMobject("Output")
        output_label.shift(2*RIGHT + 3*UP)
        output_label.scale(0.5)
        hidden_label = TextMobject("Hidden")
        hidden_label.shift(3*UP)
        hidden_label.scale(0.5)
        input_label = TextMobject("Input")
        input_label.shift(2*LEFT + 3*UP)
        input_label.scale(0.5)

        # Create the matrix representation of the image
        matrix = TexMobject(r"\begin{bmatrix} 0.91 & 0.08 & 0.76 \\ 0.52 & 0.95 & 0.12 \\ 0.10 & 0.88 & 0.17 \end{bmatrix}")
        matrix.shift(4 * LEFT)
        matrix.scale(0.75)

        # vector of input matrix
        vector = TexMobject(r"\begin{bmatrix} 0.91 \\ 0.08 \\ 0.76 \\ 0.52 \\ 0.95 \\ 0.12 \\ 0.10 \\ 0.88 \\ 0.17 \end{bmatrix}")
        vector.shift(4 * LEFT)
        vector.scale(0.9)

        # vector of output matrix
        vector_o = TexMobject("0.53")
        vector_o.shift(4 * RIGHT)

        # label of output verctor
        label_o = TextMobject("Is Not ACL Tear")
        label_o.shift(4 * RIGHT + 0.5 * DOWN)
        label_o.scale(0.5)

        # label of output true label
        label_t = TextMobject("True Label: ACL Tear")
        label_t.shift(4 * RIGHT + 1.5 * DOWN)
        label_t.scale(0.5)

        # label of output true label (new)
        vector_o_new = TexMobject("0.37")
        vector_o_new.shift(4 * RIGHT)

        label_o_new = TextMobject("Is ACL Tear")
        label_o_new.scale(0.5)
        label_o_new.shift(4 * RIGHT + 0.5 * DOWN)

        self.play(FadeIn(input_image))
        self.wait()
        self.play(input_image.shift, LEFT * 4)
        self.wait()

        self.play(Write(myNetwork))
        self.play(Write(output_label), Write(hidden_label), Write(input_label))
        self.wait()

        self.play(FadeOut(input_image))
        self.play(Write(matrix))
        self.wait()

        self.play(Transform(matrix, vector))
        self.wait()


        # Feed forward
        layers = myNetwork.layers
        for i in range(len(layers) - 1):
            layer1 = layers[i]
            layer2 = layers[i + 1]
            params = []
            for j, neuron1 in enumerate(layer1.neurons):
                for k, neuron2 in enumerate(layer2.neurons):
                    params.append(ShowPassingFlash(myNetwork.get_edge(neuron1, neuron2).set_color(YELLOW), time_width=random.random()))
            self.play(*params)


        self.play(Write(vector_o))
        self.wait()
        self.play(Write(label_o))
        self.wait()
        self.play(Write(label_t))
        self.wait()

        # Explain gradient descent
        gradient_descent = TextMobject("Gradient Descent")
        gradient_descent.shift(3 * DOWN)
        gradient_descent.scale(0.75)
        self.play(Write(gradient_descent))


        # backwards
        layers = myNetwork.layers
        for i in range(len(layers)-1, 0, -1):
            layer1 = layers[i]
            layer2 = layers[i - 1]
            params = []
            for j, neuron1 in enumerate(layer1.neurons):
                for k, neuron2 in enumerate(layer2.neurons):
                    params.append(ShowPassingFlash(myNetwork.get_edge(neuron1, neuron2).set_color(YELLOW), time_width=random.random()))
                    params.append(Indicate(neuron2))
            self.play(*params)
        self.wait()

        # self.play(FadeOut(gradient_descent))

        # Feed forward
        layers = myNetwork.layers
        for i in range(len(layers) - 1):
            layer1 = layers[i]
            layer2 = layers[i + 1]
            params = []
            for j, neuron1 in enumerate(layer1.neurons):
                for k, neuron2 in enumerate(layer2.neurons):
                    params.append(ShowPassingFlash(myNetwork.get_edge(neuron1, neuron2).set_color(YELLOW), time_width=random.random()))
            self.play(*params)

        # Update the output
        self.play(Transform(vector_o, vector_o_new))
        self.play(Transform(label_o, label_o_new))
        self.wait()

        # clear Scene
        self.play(FadeOut(myNetwork), FadeOut(output_label), FadeOut(hidden_label), FadeOut(input_label), FadeOut(matrix), FadeOut(vector_o), FadeOut(label_o), FadeOut(label_t))

        # Move gradient descent to the top
        self.play(gradient_descent.shift, UP * 6)

        gd_formula = TexMobject(r"\theta \leftarrow \theta - \alpha \frac{\partial J}{\partial \theta}")

        # add the arrow and gd_formula to the scene
        self.play(Write(gd_formula))
        self.wait()



class AdversarialAttack(Scene):

    def construct(self):
        gradient_descent = TextMobject("Gradient Descent")
        gradient_descent.shift(3 * DOWN)
        gradient_descent.scale(0.75)
        self.play(Write(gradient_descent))
        self.play(gradient_descent.shift, UP * 6)

        # display gradient descent formula
        gd_formula = TexMobject(r"\theta \leftarrow \theta - \alpha \frac{\partial J}{\partial \theta}")

        self.play(Write(gd_formula))
        self.wait()

        # changing to adversarial attack
        adv_formula = TexMobject(r"X \leftarrow X + \epsilon \frac{\partial J}{\partial X}")
        self.play(FadeOut(gd_formula))
        self.play(Write(adv_formula))
        # self.play(Transform(gd_formula, adv_formula))

        gradient_ascent = TextMobject("Gradient Ascent on ")
        gradient_ascent.shift(3 * UP + 0.5 * LEFT)
        gradient_ascent.scale(0.75)
        self.play(FadeOut(gradient_descent))
        self.play(Write(gradient_ascent))
        # self.play(Transform(gradient_descent, gradient_ascent))

        X = TexMobject(r"X")
        X.shift(3 * UP + 1.5 * RIGHT)
        X.scale(0.75)
        X.set_color(BLUE)
        self.play(Write(X))
        self.wait()

        # intro adverasrial attack
        adv_attack = TextMobject("Fast Gradient Sign Method")
        adv_attack.shift(3 * UP)
        adv_attack.scale(0.75)

        fgsm_formula = TexMobject(r"X \leftarrow X + \epsilon sign(\frac{\partial J}{\partial X})")
        self.play(FadeOut(X), Transform(adv_formula, fgsm_formula))
        self.play(FadeOut(gradient_ascent), Write(adv_attack))

        self.wait()



class AdversarialTraining(Scene):
    def construct(self):
        # Create the Neural Network
        myNetwork = NeuralNetworkMobject([9, 4, 4, 1])
        myNetwork.label_inputs("x")
        myNetwork.label_hidden_layers("h")
        myNetwork.label_outputs("\hat{y}")

        # Create the layer label
        output_label = TextMobject("Output")
        output_label.shift(2*RIGHT + 3*UP)
        output_label.scale(0.5)
        hidden_label = TextMobject("Hidden")
        hidden_label.shift(3*UP)
        hidden_label.scale(0.5)
        input_label = TextMobject("Input")
        input_label.shift(2*LEFT + 3*UP)
        input_label.scale(0.5)

        # vector of input matrix
        vector = TexMobject(r"\begin{bmatrix} 0.91 \\ 0.08 \\ 0.76 \\ 0.52 \\ 0.95 \\ 0.12 \\ 0.10 \\ 0.88 \\ 0.17 \end{bmatrix}")
        vector.shift(4 * LEFT)
        vector.scale(0.9)
        vector_adv = TexMobject(r"\begin{bmatrix} 0.91 + 0.01 \\ 0.08 - 0.01 \\ 0.76 - 0.01 \\ 0.52 + 0.01 \\ 0.95 + 0.01 \\ 0.12 + 0.01 \\ 0.10 - 0.01 \\ 0.88 - 0.01 \\ 0.17 + 0.01 \end{bmatrix}")
        vector_adv.shift(4 * LEFT)
        vector_adv.scale(0.9)

        # vector of output matrix
        vector_o = TexMobject("0.37")
        vector_o.shift(4 * RIGHT)
        vector_o_adv = TexMobject("0.99")
        vector_o_adv.shift(4 * RIGHT)

        # label of output verctor
        label_o = TextMobject("Is ACL Tear")
        label_o.shift(4 * RIGHT + 0.5 * DOWN)
        label_o.scale(0.5)
        label_o_adv = TextMobject("Is Not ACL Tear")
        label_o_adv.shift(4 * RIGHT + 0.5 * DOWN)
        label_o_adv.scale(0.5)

        # label of output true label
        label_t = TextMobject("True Label: ACL Tear")
        label_t.shift(4 * RIGHT + 1.5 * DOWN)
        label_t.scale(0.5)

        self.play(Write(myNetwork))
        self.play(Write(output_label), Write(hidden_label), Write(input_label))
        self.wait()

        self.play(Write(vector))
        self.wait()


        # Feed forward
        layers = myNetwork.layers
        for i in range(len(layers) - 1):
            layer1 = layers[i]
            layer2 = layers[i + 1]
            params = []
            for j, neuron1 in enumerate(layer1.neurons):
                for k, neuron2 in enumerate(layer2.neurons):
                    params.append(ShowPassingFlash(myNetwork.get_edge(neuron1, neuron2).set_color(YELLOW), time_width=random.random()))
            self.play(*params)


        self.play(Write(vector_o))
        self.wait()
        self.play(Write(label_o))
        self.wait()
        self.play(Write(label_t))
        self.wait()

        # # Explain gradient descent
        gradient_descent = TextMobject("Gradient Descent")
        gradient_descent.shift(3 * DOWN)
        gradient_descent.scale(0.75)
        # self.play(Write(gradient_descent))


        # # backwards
        # layers = myNetwork.layers
        # for i in range(len(layers)-1, 0, -1):
        #     layer1 = layers[i]
        #     layer2 = layers[i - 1]
        #     params = []
        #     for j, neuron1 in enumerate(layer1.neurons):
        #         for k, neuron2 in enumerate(layer2.neurons):
        #             params.append(ShowPassingFlash(myNetwork.get_edge(neuron1, neuron2).set_color(YELLOW), time_width=random.random()))
        #             params.append(Indicate(neuron2))
        #     self.play(*params)
        # self.wait()

        self.play(FadeOut(vector), FadeIn(vector_adv))
        self.wait()
        # self.play(FadeOut(gradient_descent))

        # Feed forward
        layers = myNetwork.layers
        for i in range(len(layers) - 1):
            layer1 = layers[i]
            layer2 = layers[i + 1]
            params = []
            for j, neuron1 in enumerate(layer1.neurons):
                for k, neuron2 in enumerate(layer2.neurons):
                    params.append(ShowPassingFlash(myNetwork.get_edge(neuron1, neuron2).set_color(YELLOW), time_width=random.random()))
            self.play(*params)

        self.play(Transform(vector_o, vector_o_adv), Transform(label_o, label_o_adv))
        self.wait()

        # Backwards on y_true of X_adv
        label_t = TextMobject("True Label: ACL Tear")
        label_t.shift(4 * RIGHT + 1.5 * DOWN)
        label_t.scale(0.5)
        self.play(Write(label_t))
        self.wait()


        self.play(FadeIn(gradient_descent))
        # backwards
        layers = myNetwork.layers
        for i in range(len(layers)-1, 0, -1):
            layer1 = layers[i]
            layer2 = layers[i - 1]
            params = []
            for j, neuron1 in enumerate(layer1.neurons):
                for k, neuron2 in enumerate(layer2.neurons):
                    params.append(ShowPassingFlash(myNetwork.get_edge(neuron1, neuron2).set_color(YELLOW), time_width=random.random()))
                    params.append(Indicate(neuron2))
            self.play(*params)
        self.wait()
        self.play(FadeOut(gradient_descent))
        self.play(FadeOut(vector_o_adv), FadeOut(label_o_adv), FadeOut(vector_o), FadeOut(label_o))

        vector_o_adv_ed = TextMobject("0.42")
        label_o_adv_ed = TextMobject("Is ACL Tear")

        vector_o_adv_ed.shift(4 * RIGHT)
        label_o_adv_ed.shift(4 * RIGHT + 0.5 * DOWN)
        label_o_adv_ed.scale(0.5)

        self.play(FadeIn(vector_o_adv_ed), FadeIn(label_o_adv_ed))

class Summary(Scene):
    def construct(self):
        title = TextMobject("To Sum Up")
        title.scale(1)
        title.shift(3 * UP)
        self.play(Write(title))
        self.wait()

        formula_generate_perturbation = TexMobject("\delta^{FGSM}_i \leftarrow \epsilon sign(\nabla_{X_i}J(\thata, X_i, y_i))")
        formula_generate_perturbation.scale(0.75)
        formula_generate_perturbation.shift(2.5 * DOWN + 2 * LEFT)

        formula_perturb_input = TexMobject("X^{adv}_i \leftarrow X_i + \delta^{FGSM}")
        formula_perturb_input.scale(0.75)
        formula_perturb_input.shift(2.5 * DOWN)

        formula_backprop = TexMobject("\theta \leftarrow \theta - \alpha \nabla_{\theta}J(\theta, X^{adv}_i, y_i)")
        formula_backprop.scale(0.75)
        formula_backprop.shift(2.5 * DOWN + 2 * RIGHT)

        self.play(Write(formula_generate_perturbation))
        self.wait()
        self.play(Write(formula_perturb_input))
        self.wait()
        self.play(Write(formula_backprop))
        self.wait()



# A customizable Sequential Neural Network
class NeuralNetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius": 0.15,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": LARGE_BUFF,
        "output_neuron_color": YELLOW,
        "input_neuron_color": WHITE,
        "hidden_layer_neuron_color": WHITE,
        "neuron_stroke_width": 2,
        "neuron_fill_color": GREEN,
        "edge_color": LIGHT_GREY,
        "edge_stroke_width": 2,
        "edge_propogation_color": YELLOW,
        "edge_propogation_time": 0.02,
        "max_shown_neurons": 16,
        "brace_for_large_layers": True,
        "average_shown_activation_of_large_layer": True,
        "include_output_labels": False,
        "arrow": False,
        "arrow_tip_size": 0.1,
        "left_size": 1,
        "neuron_fill_opacity": 1
    }
    # Constructor with parameters of the neurons in a list
    def __init__(self, neural_network, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layer_sizes = neural_network
        self.add_neurons()
        self.add_edges()
        self.add_to_back(self.layers)
    # Helper method for constructor
    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size, index)
            for index, size in enumerate(self.layer_sizes)
        ])
        layers.arrange_submobjects(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        if self.include_output_labels:
            self.label_outputs_text()
    # Helper method for constructor
    def get_nn_fill_color(self, index):
        if index == -1 or index == len(self.layer_sizes) - 1:
            return self.output_neuron_color
        if index == 0:
            return self.input_neuron_color
        else:
            return self.hidden_layer_neuron_color
    # Helper method for constructor
    def get_layer(self, size, index=-1):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Circle(
                radius=self.neuron_radius,
                stroke_color=self.get_nn_fill_color(index),
                stroke_width=self.neuron_stroke_width,
                fill_color=BLACK,
                fill_opacity=self.neuron_fill_opacity,
            )
            for x in range(n_neurons)
        ])
        neurons.arrange_submobjects(
            DOWN, buff=self.neuron_to_neuron_buff
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = TexMobject("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer
    # Helper method for constructor

    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)
    # Helper method for constructor
    def get_edge(self, neuron1, neuron2):
        if self.arrow:
            return Arrow(
                neuron1.get_center(),
                neuron2.get_center(),
                buff=self.neuron_radius,
                stroke_color=self.edge_color,
                stroke_width=self.edge_stroke_width,
                tip_length=self.arrow_tip_size
            )
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.neuron_radius,
            stroke_color=self.edge_color,
            stroke_width=self.edge_stroke_width,
        )

    # Labels each input neuron with a char l or a LaTeX character
    def label_inputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[0].neurons):
            label = TexMobject(f"{l}_"+"{"+f"{n + 1}"+"}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels each output neuron with a char l or a LaTeX character
    def label_outputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = TexMobject(f"{l}_"+"{"+f"{n + 1}"+"}")
            label.set_height(0.4 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels each neuron in the output layer with text according to an output list
    def label_outputs_text(self, outputs):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = TexMobject(outputs[n])
            label.set_height(0.75*neuron.get_height())
            label.move_to(neuron)
            label.shift((neuron.get_width() + label.get_width()/2)*RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels the hidden layers with a char l or a LaTeX character
    def label_hidden_layers(self, l):
        self.output_labels = VGroup()
        cnt = 1
        for layer in self.layers[1:-1]:
            for n, neuron in enumerate(layer.neurons):
                label = TexMobject(f"{l}_{cnt}")
                cnt += 1
                label.set_height(0.4 * neuron.get_height())
                label.move_to(neuron)
                self.output_labels.add(label)
        self.add(self.output_labels)
