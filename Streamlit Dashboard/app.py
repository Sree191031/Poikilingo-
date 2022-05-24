import numpy as np
import io
import streamlit as st
from streamlit_pages.streamlit_pages import MultiPage
import os
from collections import Counter
import random
import pandas as pd
import plotly.express as px
from typing import List, Tuple, Optional
import turicreate as tc
from auth import session
from auth.password import with_password
from dotenv import load_dotenv

# load model once
from model import trending_model, collaborative_model


session_state = session.get(password=False)

load_dotenv()

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


@with_password(session_state)
def main():

    # st.sidebar.markdown(
    #     """
    #     Poikilingo Recommender System Dashboard
    #     """
    # )

    st.sidebar.title("Control Panel")

    display = ("Trending Based Recommendation", "Collaborative Filtering Recommendation", "Content Based Recommendation", "Hybrid Recommendation")

    options = list(range(len(display)))

    recommenders_system_titles = lambda x: display[x]

    with st.sidebar:
    
        # Used for visualizing data is csv files
        with st.form(key="upload_csv"):
            st.header("Upload Data")
            uploaded_files = st.file_uploader(label="Upload CSV ", accept_multiple_files=False, type='csv')
            upload_csv = st.form_submit_button(label="Upload")

        recommender_system_option = st.selectbox('Select an algorithm.', options, format_func=recommenders_system_titles, key='algorithm_option')

        # Used for looking some model outputs
        with st.form('Form'):
            if st.session_state.algorithm_option in [0, 1]:
                user = st.number_input('Select user', step=1)
            n_activities = st.slider(label='Select number of activities to recommend', min_value=0, max_value=100, key="4")
            submitted = st.form_submit_button('Submit')

    # function which takes saved model file, user id for which recommendations need to done
    # and num of recommendations as parameter and returns the recommendations as output
    def make_recommendations(model_tc, user_id, num_recommendations):
        user_recomm = model_tc.recommend(users=[user_id], k=num_recommendations)
        df_user_recomm = user_recomm.to_dataframe()
        return df_user_recomm.set_index('rank', drop=False)

    st.title("Poikilingo Recommender System Dashboard")

    student_activity_df = None
    if upload_csv and uploaded_files:
        data = uploaded_files.read()
        fp = io.BytesIO(data)
        student_activity_df = pd.read_csv(fp)
        st.header("student_activity_df data")
        st.dataframe(student_activity_df)

    if submitted:
        st.title("Top {} {}".format(n_activities, recommenders_system_titles(recommender_system_option)))

        if recommender_system_option == 0:
            # task-2 trending activities model loaded in the streamlit app, which can be used to perform prediction on demand
            # and showing the inference results
            trending_model_outputs = make_recommendations(trending_model, user, n_activities)
            st.header("model outputs")
            st.dataframe(trending_model_outputs)

        if recommender_system_option == 1:
            # task-2 matrix factorization updated model version 2 dataset loaded in the streamlit app, which can be used to perform prediction on demand
            # and showing the inference results
            collaborative_model_outputs = make_recommendations(collaborative_model, user, n_activities)
            st.header("model outputs")
            st.dataframe(collaborative_model_outputs)


    def visualize_activities(st_obj, data: Tuple[List[str], Optional[List[float]]], title, **kwargs):
        """Plot the distribution of a list of student activities.

        Args:
            data (Tuple(List[string], Optional[List[float]])): a list of activity IDs (string)
        """

        col_name = "student activities"

        if "age" in kwargs:
            age = kwargs["age"]
            col_name = f"student activities in {age} age group"

        values = None
        if len(data) == 1:
            activities, = data
        else:
            activities, values = data

        if not values:
            df = pd.DataFrame(np.array(activities)[:, np.newaxis], columns=[col_name])
        else:
            # group by the activity ids, and take the average in each group
            df = pd.DataFrame(
                {col_name: np.array(activities), "value": np.array(values)},
            )
            df = df.groupby(col_name, as_index=False).mean()

        if not values:
            fig = px.histogram(df, x=col_name, category_orders={col_name: sorted(list(set(activities)))})
        else:
            fig = px.bar(df, x=col_name, y="value")

        # Designing the interface
        # st_obj.title(title)  # title not centered
        st_obj.markdown(f"<h1 style='text-align: center; color: black;'>{title}</h1>", unsafe_allow_html=True)  # title centered

        # For newline
        st_obj.plotly_chart(fig, use_container_width=True)


    def page_1():

        def fetch_data_and_visualize_activities(st_obj, title):
            """Fetch `student_activity` and visualize activities (with histogram) on the dashboard"""

            def fetch_data():

                data = None
                if student_activity_df is not None:
                    # use the provided data
                    activities = student_activity_df['Activity_ID'].tolist()                     

                    data = (activities,)

                return data

            data = fetch_data()
            if data: 
                visualize_activities(st_obj, data, title)

        def fetch_data_and_visualize_activities_with_game_play_time(st_obj, title):
            """Fetch `student_activity` and visualize activities with `GamePlayTime` on the dashboard"""

            def fetch_data():

                data = None
                if student_activity_df is not None:
                    # use the provided data
                    grouped = student_activity_df.groupby(['Activity_ID']).mean().reset_index()
                    activities = grouped['Activity_ID'].tolist()
                    values = grouped['GamePlayTime'].tolist()                     
                    
                    data = activities, values

                return data

            data = fetch_data()
            if data: 
                visualize_activities(st_obj, data, title)

        def fetch_data_and_visualize_activities_with_translanguage_level(st_obj, title):
            """Fetch `student_activity` and visualize activities with `Translanguage_Level_2` on the dashboard"""

            def fetch_data():

                data = None
                if student_activity_df is not None:
                    # use the provided data
                    grouped = student_activity_df.groupby(['Activity_ID']).mean().reset_index()
                    activities = grouped['Activity_ID'].tolist()
                    values = grouped['Translanguage_Level_2'].tolist()                     
                    
                    data = activities, values

                return data

            data = fetch_data()
            if data: 
                visualize_activities(st_obj, data, title)

        def fetch_data_and_visualize_activities_with_usefulness_score(st_obj, title):
            """Fetch `student_activity` and visualize activities with `Usefulness` on the dashboard"""

            def fetch_data():

                data = None
                if student_activity_df is not None:
                    # use the provided data
                    grouped = student_activity_df.groupby(['Activity_ID']).mean().reset_index()
                    activities = grouped['Activity_ID'].tolist()
                    values = grouped['Usefulness'].tolist()                     
                    
                    data = activities, values

                return data

            data = fetch_data()
            if data: 
                visualize_activities(st_obj, data, title)

        # row 1
        col1, col2 = st.columns(2)
        fetch_data_and_visualize_activities(col1, title="most frequent recommended activities")
        fetch_data_and_visualize_activities_with_game_play_time(col2, title="game time on activities")

        # row 2
        col1, col2 = st.columns(2)
        fetch_data_and_visualize_activities_with_translanguage_level(col1, title="average translanguage level on activities")
        fetch_data_and_visualize_activities_with_usefulness_score(col2, title="average usefulness scores on activities")

    def page_2():

        def fetch_data_and_visualize_activities_with_translanguage_level(st_obj, title):
            """Fetch `student_activity` and visualize activities with `Translanguage_Level_2` on the dashboard"""

            def fetch_data():

                data = None
                if student_activity_df is not None:
                    # use the provided data
                    grouped = student_activity_df.groupby(['Activity_ID']).mean().reset_index()
                    activities = grouped['Activity_ID'].tolist()
                    values = grouped['Translanguage_Level_2'].tolist()                     
                    
                    data = activities, values

                return data

            data = fetch_data()
            if data: 
                visualize_activities(st_obj, data, title, age=4)

        def fetch_data_and_visualize_activities_with_usefulness_score(st_obj, title):
            """Fetch `student_activity` and visualize activities with `Usefulness` on the dashboard"""

            def fetch_data():

                data = None
                if student_activity_df is not None:
                    # use the provided data
                    grouped = student_activity_df.groupby(['Activity_ID']).mean().reset_index()
                    activities = grouped['Activity_ID'].tolist()
                    values = grouped['Usefulness'].tolist()                     
                    
                    data = activities, values

                return data

            data = fetch_data()
            if data: 
                visualize_activities(st_obj, data, title, age=4)

        # row 1
        col1, col2 = st.columns(2)
        fetch_data_and_visualize_activities_with_translanguage_level(col1, title="average translanguage level on activities")
        fetch_data_and_visualize_activities_with_usefulness_score(col2, title="average usefulness scores on activities")


    # call app class object
    app = MultiPage()
    # Add pages
    app.add_page("Page 1", page_1)
    app.add_page("Page 2", page_2)
    app.run()


if __name__ == '__main__':
    main()
