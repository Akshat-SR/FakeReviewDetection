#LIBRARIES
import streamlit as st
import pickle
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer 
import re


#LOAD PICKLE FILES
model = pickle.load(open('D:/Fake Review Detection/Project/data and pickle files/best_model.pkl','rb'))
vectorizer = pickle.load(open('D:/Fake Review Detection/Project/data and pickle files/count_vectorizer.pkl','rb'))
model_mnb = pickle.load(open('D:/Fake Review Detection/Project/data and pickle files/mnb_model.pkl','rb'))
vectorizer_mnb = pickle.load(open('D:/Fake Review Detection/Project/data and pickle files/count_vectorizer_mnb.pkl','rb'))
model_svm = pickle.load(open('D:/Fake Review Detection/Project/data and pickle files/svm_model.pkl','rb'))
vectorizer_svm = pickle.load(open('D:/Fake Review Detection/Project/data and pickle files/count_vectorizer_svm.pkl','rb'))


#FOR STREAMLIT
nltk.download('stopwords')

#TEXT PREPROCESSING
sw = set(stopwords.words('english'))
def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    
    cleaned = []
    stemmed = []
    
    for token in tokens:
        if token not in sw:
            cleaned.append(token)
            
    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)

#TEXT CLASSIFICATION
def text_classification(text, selected_model):
    if len(text) < 1:
        st.write("  ")
    else:
        with st.spinner("Classification in progress..."):
            cleaned_review = text_preprocessing(text)
        if selected_model == "Logistic Regression":
            process = vectorizer.transform([cleaned_review]).toarray()    
            prediction = model.predict(process)
        elif selected_model == "Support Vector Machine":
            process = vectorizer_svm.transform([cleaned_review]).toarray()
            prediction = model_svm.predict(process)
        elif selected_model == "Multinomial Naive Bayes":
            process = vectorizer_mnb.transform([cleaned_review]).toarray()
            prediction = model_mnb.predict(process)

        p = ''.join(str(i) for i in prediction)
        if p == 'True':
                st.success("The review entered is Legitimate.")
        if p == 'False':
                st.error("The review entered is Fraudulent.")

#PAGE FORMATTING AND APPLICATION
def main():
    st.title("Fake Reviews Detection Using Machine Learning")
    
    
    # --EXPANDERS--    
    abstract = st.expander("Abstract")
    if abstract:
        abstract.write("Customers and businesses alike consider reviews to be quite helpful in today's society. It should come as no surprise that review fraud has diminished the value of the entire encounter—from violating international laws to leaving unfavourable evaluations that damage the company's reputation. Since this issue is linked to natural language processing, it has been identified as an emerging topic. Therefore, in order to make significant progress in this field, a variety of machine learning approaches and techniques have to be developed. Numerous online retailers, including Amazon, have implemented processes such as Verified Purchase, which certifies the accuracy of the review language when an item is bought straight from the website. This work suggests training three classifiers for supervised training on Amazon's labelled dataset using the verified purchases label. The classifiers used were MNB, SVM, and LR. Two different vectorizers, Count Vectorizer and TF-IDF Vectorizers, were used to tune the model. All trained models achieved an accuracy rate of 80% overall, suggesting that the vectorizers performed well and that real and fake reviews can be distinguished. The models' performance was enhanced more by the count vectorizer of the two, and LR outperformed the other three inside counts, with an accuracy rate of 85% and a recall rate of 92%.")
        #st.write(abstract)
    
    
    #--MODEL SELECTION--
    selected_model = st.selectbox("Select Model:", ["Logistic Regression", "Support Vector Machine", "Multinomial Naive Bayes"])
    
    # --CHECKBOXES--
    st.subheader("Information on the Classifier")
    if st.checkbox("About Classifier"):
        if  selected_model == "Logistic Regression":
            st.markdown('**Model:** Logistic Regression')
            st.markdown('**Vectorizer:** Count')
            st.markdown('**Test-Train splitting:** 40% - 60%')
            st.markdown('**Spelling Correction Library:** TextBlob')
            st.markdown('**Stemmer:** PorterStemmer')
        elif selected_model == "Support Vector Machine":
            st.markdown('**Model:** Support Vector Machine')
            st.markdown('**Vectorizer:** Count')
            st.markdown('**Test-Train splitting:** 40% - 60%')
            st.markdown('**Spelling Correction Library:** TextBlob')
            st.markdown('**Stemmer:** PorterStemmer') 
        elif selected_model == "Multinomial Naive Bayes":
            st.markdown('**Model:** Multinomial Naive Bayes')
            st.markdown('**Vectorizer:** Count')
            st.markdown('**Test-Train splitting:** 40% - 60%')
            st.markdown('**Spelling Correction Library:** TextBlob')
            st.markdown('**Stemmer:** PorterStemmer')

        
    if st.checkbox("Evaluation Results"):
        if  selected_model == "Logistic Regression":
                st.markdown('**Model:** Logistic Regression')
                st.markdown('**Accuracy:** 85%')
                st.markdown('**Precision:** 80%')
                st.markdown('**Recall:** 92%')
                st.markdown('**F-1 Score:** 85%')
        elif  selected_model == "Support Vector Machine":
                st.markdown('**Model:** Support Vector Machine')
                st.markdown('**Accuracy:** 84%')
                st.markdown('**Precision:** 79%')
                st.markdown('**Recall:** 91%')
                st.markdown('**F-1 Score:** 85%')
        elif  selected_model == "Multinomial Naive Bayes":
                st.markdown('**Model:** Multinomial Naive Bayes')
                st.markdown('**Accuracy:** 80%')
                st.markdown('**Precision:** 81%')
                st.markdown('**Recall:** 77%')
                st.markdown('**F-1 Score:** 79%')


    #--IMPLEMENTATION OF THE CLASSIFIER--
    st.subheader("Fake Review Classifier")
    review = st.text_area("Enter Review: ")
    if st.button("Check"):
        text_classification(review, selected_model)

#RUN MAIN        
main()