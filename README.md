# RAMI Project: Molecular Structure Generation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Features](#features)
5. [Dependencies](#dependencies)
6. [Configuration](#configuration)
7. [Documentation](#documentation)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)
10. [Contributors](#contributors)
11. [License](#license)

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



## Features

- **Constraint-Based Generation**: Harnesses constraint programming to ensure the generation of chemically valid structures.
- **JMol Integration**: Utilizes JMol for sophisticated 3D visualization of the molecular structures.
- **Automated Instance & Experience Creation**: Features classes like `InstanceMaker` and `ExperienceMaker` for automated generation and analysis of molecular structures.
- **CML Output**: Structures are output in Chemical Markup Language (CML), ensuring compatibility with a wide range of molecular visualization tools.

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
