# RAMI Project: Molecular Structure Generation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Code](#code).
5. [Features](#features)
6. [Dependencies](#dependencies)
7. [Configuration](#configuration)
8. [Documentation](#documentation)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)
11. [Contributors](#contributors)
12. [License](#license)

## Introduction

This project, orchestrated by Nicolas Prcovic and realized by Manon Girard, Romain Durand, and Paul Peyssard, delves into the realm of automatic molecular structure generation. Originating from Adrien Varet's innovative work on benzene structures, this software transcends its origins to provide a comprehensive solution for generating molecular structures from any given list of atoms. At its core, it harnesses the power of constraint programming, forgoing the need for a graphical interface, to generate structures and collaborates with JMol for their visualization.

## Installation

Follow these steps to set up the project:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repository/rami-project.git
    ```

2. **Install Dependencies**:
    Navigate to the project directory and install the necessary dependencies.
    ```bash
    cd rami-project
    pip install -r requirements.txt
    ```

## Usage

Mettre comment utiliser le programme (à la fin)

## Features

- **Constraint-Based Generation**: Harnesses constraint programming to ensure the generation of chemically valid structures.
- **JMol Integration**: Utilizes JMol for sophisticated 3D visualization of the molecular structures.
- **Automated Instance & Experience Creation**: Features classes like `InstanceMaker` and `ExperienceMaker` for automated generation and analysis of molecular structures.
- **CML Output**: Structures are output in Chemical Markup Language (CML), ensuring compatibility with a wide range of molecular visualization tools.

## Code

This section provides a detailed overview of each class in the project, explaining their purpose and functionality within the molecular structure generation system.

### Atom.java

**Purpose**: Represents an individual atom in a molecule.
- **Key Attributes**:
  - `type`: Specifies the type of the atom (e.g., Carbon, Hydrogen).
  - `bonds`: A collection of bonds that this atom forms with other atoms.
- **Key Methods**:
  - `addBond()`: Adds a bond to another atom.
  - `validateValency()`: Ensures that the atom's valency rules are not violated.

### AtomIndexer.java

**Purpose**: Manages the indexing and retrieval of atoms in a molecule.
- **Key Methods**:
  - `getIndex(Atom atom)`: Retrieves the index of a given atom.
  - `getAtom(int index)`: Retrieves an atom based on its index.

### BondDistance.java

**Purpose**: Stores and manages permissible distances between different types of atoms.
- **Key Methods**:
  - `getMinDistance(AtomType a, AtomType b)`: Retrieves the minimum permissible distance between two atom types.
  - `getMaxDistance(AtomType a, AtomType b)`: Retrieves the maximum permissible distance between two atom types.

### CML_generator.java

**Purpose**: Converts the internal molecular structure representation into Chemical Markup Language (CML) format.
- **Key Methods**:
  - `generateCML()`: Generates a CML representation of the current molecular structure.

### GraphModelisation.java

**Purpose**: Represents the molecular structure as a graph.
- **Key Methods**:
  - `addAtom(Atom atom)`: Adds an atom to the graph.
  - `addBond(Atom a, Atom b)`: Adds a bond between two atoms in the graph.

### Main.java

**Purpose**: Serves as the entry point of the application.
- **Key Methods**:
  - `main()`: Initializes the application and starts the molecular structure generation process.

### MainViz.java

**Purpose**: Provides visualization capabilities for the generated molecular structures.
- **Key Methods**:
  - `visualizeStructure()`: Generates a visual representation of the molecular structure.

### Modelisation.java

**Purpose**: Handles the logic for modeling the molecular structure based on given constraints.
- **Key Methods**:
  - `modelMolecule()`: Models a molecule based on specified constraints and atom types.

### MoleculeUtils.java

**Purpose**: Provides utility functions for operations on molecules.
- **Key Methods**:
  - `calculateMolecularWeight()`: Calculates the molecular weight of the given molecule.

### InstanceMaker.java

**Purpose**: Automates the creation of molecular structure instances for testing and analysis.
- **Key Methods**:
  - `generateInstance()`: Generates a new instance of a molecular structure based on specified parameters.

### ExperienceMaker.java

**Purpose**: Facilitates the execution of multiple instances of molecular structure generation for performance analysis and testing.
- **Key Methods**:
  - `runExperiences()`: Runs multiple instances of molecular structure generation and aggregates the results.

Each class is designed to encapsulate specific functionality and to interact with other classes in a cohesive manner, ensuring a modular and maintainable codebase.


## Dependencies

The project requires the following dependencies:

- Java Development Kit (JDK) for running Java applications.
- JMol for molecular structure visualization.
- Choco Solver for constraint programming.

Ensure these are properly installed and configured before running the project.

## Configuration

The project can be configured through the `config.json` file, where parameters like the type and quantity of atoms, as well as structural constraints, can be specified.

## Documentation

The project documentation is available in the `docs` folder. It provides detailed information about the project's structure, classes, and methods, as well as usage examples and troubleshooting tips.

## Examples

Refer to the `examples` folder for sample input files and their corresponding output structures. This can provide a good starting point for understanding how to structure your input files.

## Troubleshooting

If you encounter issues, please check the `logs` folder for error messages and stack traces. Common issues and their resolutions are documented in the `TROUBLESHOOTING.md` file.

## Contributors

- Manon Girard
- Romain Durand
- Paul Peyssard
- Supervised by Nicolas Prcovic

## License

This project is licensed under the [LICENSE NAME]. Please see the `LICENSE` file for more details.
