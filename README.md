[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ol4GAg0d)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=15423632)
<!--
Name of your teams' final project
-->
# Bird Flock Behavior Prediction Using Deep Learning Techniques
## [National Action Council for Minorities in Engineering(NACME)](https://www.nacme.org) Artificial Intelligence - Machine Learning (AIML) Intensive Summer Bootcamp at the [University of Southern California](https://viterbischool.usc.edu)

<!--
List all of the members who developed the project and
link to each members respective GitHub profile
-->
Developed by: 
- [Parker Sepulvado](https://github.com/Parkersep) - `Mechanical Engineering` - `Texas A&M University`
- [Camile Reese](https://github.com/creese04) - `Mechanical Engineering` - `North Carolina Agricultural & Technical State University` 
- [Davis Adams](https://github.com/davissadams) - `Mechanical Engineering` - `Cornell University` 
- [Jeffrey Milan](https://github.com/jmillan736) - `Electrical Engineering/Computer Science` - `University of California, Berkeley`

## Description
The project goal is to explore the intricate collective behaviors exhibited by bird flocks using advanced deep learning techniques. To achieve this, we will design a dataset with tens of thousands of examples of agents in specific formations, generated through simulation. The project involves integrating a generative multi-agent trajectory model that combines a Generative Adversarial Network (GAN) with a Graph Neural Network (GNN). The modified GAN pipeline will first generate a random graph based on selected points and compute adjacency matrices. A GNN Recurrent Neural Network (GRNN) will then produce trajectories from this graph, which will be decoded into physical agent positions using an appropriate architecture. Finally, the generated formations will be evaluated by discriminators to refine the model, aiming to learn and predict complex multi-agent behaviors through this integrated approach.

<!-- Give a short description on what your project accomplishes and what tools is uses. In addition, you can drop screenshots directly into your README file to add them to your README. Take these from your presentations.
-->
## Bird Flock Behavior Prediction Using Boid Simulation
Bird flock behavior prediction involves understanding and modeling how birds move collectively in formations such as v-shapes or diamond-shapes. These formations are a result of complex interactions between individual birds, and modeling these behaviors can provide insights into the underlying rules governing such collective movements using boid simulation. The boid simulation models the behavior of birds based on three simple rules: separation, alignment, and cohesion. Modeling these behaviors can provide insights into the underlying rules governing such collective movements. We then modeled our boid simulation to behave more in the way birds would behave. By using a nearest neighbor function when the cohesion distance is too far and adding in a emo_birds function which at every timestep makes some birds split from the pack.

https://github.com/user-attachments/assets/9f82b806-0196-40b0-8f9e-5937270d90a5

<!-- <img width="720" alt="Screenshot 2024-07-23 at 10 53 10 AM" src="https://github.com/user-attachments/assets/f46f280b-a5d9-4ffa-b541-440f3c6a4982"> -->

## Our Goals for This Project
1. Design a dataset with examples of agents in specific formations using a boid simulation.
1. Integrate a Generative Adversarial Network (GAN) with a Graph Neural Network (GNN) to model bird flock behaviors.
1. Generate random graphs and compute adjacency matrices for the formations.
1. Use a GNN Recurrent Neural Network (GRNN) to produce trajectories from the graphs.
1. Decode the trajectories into physical agent positions.
1. Refine the model using discriminators to improve the accuracy of the generated formations.
1. Learn and predict complex multi-agent behaviors through this integrated approach.

![image](https://github.com/user-attachments/assets/53df6242-06ee-47e4-96bb-75ae8b477bb0)
![image](https://github.com/user-attachments/assets/efb113cb-487a-4fa7-b7e6-6cb62a408a71)

## Models
1. Boid Simulation: Simulates the behavior of birds based on separation, alignment, and cohesion rules.
1. Generative Adversarial Network (GAN): Generates random graphs and computes adjacency matrices.
1. Graph Neural Network (GNN) Recurrent Neural Network (GRNN): Produces trajectories from the generated graphs.
1. Discriminators: Evaluate and refine the generated formations to improve model accuracy.
1. We will experiment with different architectures and hyperparameters to optimize the performance of these models.




## Usage instructions
<!--
Give details on how to install fork and install your project. You can get all of the python dependencies for your project by typing `pip3 freeze requirements.txt` on the system that runs your project. Add the generated `requirements.txt` to this repo.
-->
1. Fork this repo
2. Change directories into your project
3. On the command line, type `pip3 install requirements.txt`
4. On the command line, type `pip install os-sys`
5. On the command line, type `pip install numpy`
6. On the command line, type `pip install pandas`
7. Locate and open the file you downloaded from this repo

## Resources
1. Hugging Face. “Generative Adversarial Networks (GANs).” [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course/en/unit5/generative-models/gans). Accessed 23 July 2024.
   
1. Liang, Chen, et al. "Graph Neural Networks and their Applications." [arXiv](https://arxiv.org/abs/2110.11401). 21 October 2021.

1. Liang, Chen, et al. "Graph Neural Networks and their Applications." [arXiv PDF](https://arxiv.org/pdf/2110.11401). 21 October 2021.

1. Alelab UPenn. "Graph Neural Networks." [GitHub](https://github.com/alelab-upenn/graph-neural-networks). Accessed 23 July 2024.

1. Chen, Xi, et al. "Modeling and Analyzing the Dynamics of Birds' Flock Behavior." [World Scientific](https://www.worldscientific.com/doi/epdf/10.1142/S0217979216500028). 2016.

1. Eater. "Boids: An Interactive Simulation." [Eater.net](https://eater.net/boids). Accessed 23 July 2024.

1. Reynolds, Craig W. “Flocks, Herds, and Schools: A Distributed Behavioral Model.” [Inria](https://team.inria.fr/imagine/files/2014/10/flocks-hers-and-schools.pdf). 1987.

1. PyTorch Geometric. "Introduction to PyTorch Geometric." [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html). Accessed 23 July 2024.

1. PyTorch Geometric. "Data Cheatsheet." [PyTorch Geometric Cheatsheet](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/data_cheatsheet.html). Accessed 23 July 2024.

1. PyTorch Geometric Temporal. "Introduction to PyTorch Geometric Temporal." [PyTorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/index.html). Accessed 23 July 2024.

1. Deep Findr. “Introduction to GANs.” [YouTube](https://www.youtube.com/watch?v=WEWq93tioC4&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z&index=18&ab_channel=DeepFindr). Accessed 23 July 2024.

1. Deep Findr. “Understanding GANs.” [YouTube](https://www.youtube.com/watch?v=Rws9mf1aWUs&ab_channel=DeepFindr). Accessed 23 July 2024.

1. PyTorch. “DCGAN Tutorial: Face Generation.” [PyTorch Tutorials](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html). Accessed 23 July 2024.


## Questions
Please feel free to contact

Davis Adams: davis.adams441@gmail.com

Camile Reese: camilereese04@gmail.com

Parker Sepulvado: parkerlsepulvado@gmail.com

Jeffrey Millan: jmillan736@berkeley.edu
