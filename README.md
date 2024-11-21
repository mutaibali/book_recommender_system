# 📚 Book Recommendation System

This project builds a **Content-Based Book Recommendation System** using book ratings and metadata. By leveraging user ratings and book features such as titles, authors, and genres, the system recommends similar books based on cosine similarity. The project demonstrates data preprocessing, filtering, and machine learning techniques to provide meaningful recommendations.

---

## 📌 Project Overview

- **Purpose**: To recommend books similar to a given book based on user ratings and metadata.
- **Core Features**:
  - Preprocess and filter book data to identify popular and frequently rated books.
  - Generate recommendations using **Cosine Similarity** on book ratings.
  - Store preprocessed data for deployment in a web application.

---

## 🛠️ Skills Demonstrated

- **Data Preprocessing**:
  - Merging datasets, handling missing values, and removing duplicates.
  - Filtering active users and frequently rated books.
- **Feature Engineering**:
  - Pivot tables to represent user ratings for books.
  - Vectorizing book ratings using cosine similarity.
- **Python Programming**:
  - Libraries: `pandas`, `numpy`, `scikit-learn`, `pickle`.
- **Deployment Preparation**:
  - Save processed data and models using `pickle` for later use.

---

## 📂 Datasets

1. **books.csv**: Contains metadata about books (e.g., title, author, and ISBN).
2. **users.csv**: Information about users who rated books.
3. **ratings.csv**: Ratings of books by users.

---

## 🔍 Workflow and Key Steps

### 1. **Data Loading**
   - **Purpose**: Load the datasets into dataframes.
   - **Code**:
     ```python
     books = pd.read_csv('books.csv')
     users = pd.read_csv('users.csv')
     ratings = pd.read_csv('ratings.csv')
     ```

### 2. **Data Cleaning**
   - **Purpose**: Remove missing values and duplicates for clean analysis.
   - **Steps**:
     - Drop null values from all datasets.
     - Remove duplicate rows.
   - **Code**:
     ```python
     books.dropna(inplace=True)
     ratings.dropna(inplace=True)
     books.duplicated().sum()  # Check for duplicates
     ```

### 3. **Merging Datasets**
   - **Purpose**: Merge `ratings` and `books` datasets on the `ISBN` column to add book details to ratings.
   - **Code**:
     ```python
     ratings_with_name = ratings.merge(books, on='ISBN')
     ```

### 4. **Aggregating Ratings**
   - **Purpose**: Calculate the number and average rating for each book.
   - **Code**:
     ```python
     num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
     num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

     avg_rating_df = ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
     avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
     ```

### 5. **Filtering Popular Books**
   - **Purpose**: Filter books with at least 250 ratings and sort them by average rating.
   - **Code**:
     ```python
     popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
     popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False)
     ```

### 6. **Pivot Table Creation**
   - **Purpose**: Create a user-item matrix where rows are books, columns are users, and values are ratings.
   - **Code**:
     ```python
     pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
     pt.fillna(0, inplace=True)
     ```

### 7. **Cosine Similarity**
   - **Purpose**: Compute similarity between books using the user-item matrix.
   - **Code**:
     ```python
     from sklearn.metrics.pairwise import cosine_similarity
     similarity_scores = cosine_similarity(pt)
     ```

### 8. **Recommendation Function**
   - **Purpose**: Recommend the top 4 similar books for a given book title.
   - **Code**:
     ```python
     def recommend(book_name):
         index = np.where(pt.index == book_name)[0][0]
         similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
         
         data = []
         for i in similar_items:
             item = []
             temp_df = books[books['Book-Title'] == pt.index[i[0]]]
             item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
             item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
             item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
             data.append(item)
         
         return data
     ```

---

## 🚀 How to Use

1. **Set Up**:
   - Download the datasets: `books.csv`, `users.csv`, and `ratings.csv`.
   - Install required Python libraries: `pandas`, `numpy`, `scikit-learn`, `pickle`.

2. **Run the Script**:
   - Execute the script to load and process the data.
   - Use the `recommend()` function to get book recommendations:
     ```python
     recommend('The Catcher in the Rye')
     ```

3. **Example Output**:
  - To Kill a Mockingbird - Harper Lee
  -  1984 - George Orwell
  -  Brave New World - Aldous Huxley
  -  Fahrenheit 451 - Ray Bradbury


---

## 🔮 Future Enhancements

- Incorporate user demographics for a **Hybrid Recommendation System**.
- Integrate **Deep Learning** models for more robust recommendations.
- Create a web application using Flask or Streamlit for easy interaction.

---

## 🤝 Connect

Feel free to reach out for questions or suggestions:
- **Email**: [syedmutaib0599@gmail.com](mailto:syedmutaib0599@gmail.com)
- **LinkedIn**: [Syed Mutaib Ali](https://linkedin.com/in/syedmutaibali)

---

This project showcases how machine learning and data preprocessing can create a functional and interactive book recommendation system. Explore and enhance it further! 😊
