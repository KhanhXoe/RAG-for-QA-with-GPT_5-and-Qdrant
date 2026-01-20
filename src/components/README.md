# RAG Components

Core components that make up the RAG workflow pipeline. The component abstraction is designed to be modular and infinitely extensible - sharing an extremely basic contract. You can both extend existing components and create your own.

## Base Components:

### 1. Request Router
Determines how to handle incoming queries along pre-defined routes. This is where you should decide if you should answer the query and if so, send it to the appropriate route.

### 2. Query Reformulator
Reformulate and enrich the user search query. By default, we formulate it to better match vectorized document chunks and generate keywords that could be relevant.

### 3. Retriever
Retrieves documents from the vector database.

### 4. Completion Checker
Checks if the query can be feasiblt answered with the retrieved documents.

### 5. Answer Generator
Generates an answer to the query using the retrieved documents.

Note: This is just a basic set of components. You will almost certainly need to extend and/or add components to make your agentic workflow work for you. 

## Extending Components:

You can extend existing components by inheriting from them and overriding the `_execute` method. Each component comes with a base class that you can inherit from and a default implementation of the `_execute` method to take inspiration from. While standard logging is implemented, you can add your own logging by accessing the `metadata` attribute of the component.

Examples of this include creating a custom retriever that uses a different vector database. Same purpose, different approach.

## Creating New Components:
You can also add your own components by inheriting from `BaseComponent` and implementing the `_execute` method. 
For example, you could create a component that transforms a user query into a SQL query to retrieve context. You could also create components that have no AI involved and are purely software based.

Different purpose, new approach.

