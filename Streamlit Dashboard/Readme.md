# Task 5: Dashboard Building

## Description
The objective of this task was to building a dashboard to visualize (& control) the recommendation system.

## Installation
To get this dashboard up and running locally on your computer:

* You can download the zip or clone the repo with git.
* Some libraries in `requirements.txt` needed Python version < 3.9
* Create a new environment named streamlit-app, install Python 3.8, e.g. command `conda create --name streamlit-app python=3.8`.
* Activate the new environment to use it. e.g. command `conda activate streamlit-app`.
* To install all required modules use the following command in task-5 dashboard dir: `pip install -r requirements.txt`
* You need to change the filename `.env.sample` to `.env`.
* Set the password variable `DASHBOARD_PASSWORD=YOUR_PASSWORD` in `.env` file
* To run the application you can use command `streamlit run app.py`

## Password-Protected Dashboard
Password-input widget to ask user for a password and saves it as a part of Streamlit's session. The password is retrieved each time when the user interacts with the dashboard. However, each time when you refresh the dashboard or restart it, the state is lost and password should be typed again.

> Please note this password-based single-user authentication is just a simple way to somehow restrict access if running on a host with public IP.

### Screenshots
Change the filename `.env.sample` to `.env` and set the password variable `DASHBOARD_PASSWORD=YOUR_PASSWORD` in `.env` file and start streamlit as the following command shows.
```
streamlit run app.py
```

Open a link printed to the terminal with your browser. Then you'll see a password-protected dashboard as the following GIF shows.

![Authentication example](assets/auth.gif)

#### Trending Activities Model
![Trending based example](assets/trending-based.gif)

#### Collaborative Filtering Model
![Collaborative model example](assets/collaborative-model.gif)

## Streamlit Web App AWS EC2
You can access password-protected dashboard anywhere from the world using the External URL `http://3.138.105.134:8501/`. You can find the dashboard password here https://docs.google.com/document/d/1tDosI8Rj3wSIwZ0AJjxPjaqRMWQRZXvS3xfweEwKe6w/edit?usp=sharing

> Please note If you have restarted your instance, your External URL will change.

Cloud Deployment is outlined within the next sub-heading.

### Running the recommender system on a remote AWS EC2 instance

The following steps will enable you to run your recommender system on a remote EC2 instance, allowing it to the accessed by any device/application which has internet access. 

1. Ensure that you have access to a running AWS EC2 instance with an assigned public IP address.

2. Once your instance is running, select it, and copy the public IP of your instance. We will SSH into our instance. SSH allows you to control a remote machine using just your command line. Go to your command line and type the following command:

```
ssh -i <path to your key pair file> ec2-user@<your public IP>
```

It will give you another error for permission too open as your private key is not secured. To solve this problem, run this command and run the ssh command again, it will work this time.

```bash
#In Linux / Mac OSX
chmod 0400 <key pair name>
example - chmod 0400 Streamlit-app.pem
```

3. Install the required libraries - python and pip3

```bash
#Updating your instance quickly
sudo yum update -y
#Installing python3
sudo yum install python37
#Installing pip3
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
pip3 –-version
```

4. Install the prerequisite libraries that defined in `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

5. We can simply get the web app from Github by cloning it in our instance or we can transfer files from our local system using scp. 

  * Run the following commands to set up Git and clone your project. 
  
  ```bash
  #Installing git in your instance
  sudo yum install git -y
  #Checking git version
  git version
  #Cloning your repository
  git clone <your repository>
  ```
  
  * OR Using scp to transfer data. "scp” means “secure copy”, which can copy files between computers on a network. You can use this tool in a Terminal on a Unix/Linux/Mac system. To upload a file from your laptop to Amazon instance:

  ```
   scp -i ~/Desktop/amazon.pem -r ./task_5_dashboard/ ec2-user@<your public IP>:~/dashboard/
  ```
  > This command will upload all the data -  in your ~/task_5_dashboard/ folder of your laptop to folder ~/dashboard/ on an Amazon instance. Note you still need to use the private key you used to connect to the Amazon instance with ssh. (In this example, it is the amazon.pem file in ~/Desktop/.

| :information_source: NOTE :information_source:                                                                                                    |
| :--------------------                                                                                                                             |
| In the following steps we make use of the `tmux` command. This programme has many powerful functions, but for our purposes, we use it to gracefully keep our web app running in the background - even when we end our `ssh` session. You can install `tmux` using `sudo yum install tmux` |

6. Go to the dashboard directory and change the filename `.env.sample` to `.env` and set the password variable `DASHBOARD_PASSWORD=YOUR_PASSWORD` in `.env` file

7. Enter into a new Tmux session within the current directory. To do this, simply type `tmux new -s StreamlitWebApp`.  

8. Start the Streamlit web app:

```bash
streamlit run app.py
```

If this command ran successfully, output similar to the following should be observed on the Host:

```
You can now view your Streamlit app in your browser.

  Network URL: http://172.31.47.109:8501
  External URL: http://3.138.105.134:8501

```

Where the specific `Network` and `External` URLs correspond to those assigned to your own EC2 instance. Copy the value of the external URL.  

9.  Within your favourite web browser (we hope this isn't Internet Explorer 9), navigate to external URL you just copied from the Host. This should correspond to the following form:

    `http://{public-ip-address-of-remote-machine}:8501`   

    Where the above public IP address corresponds to the one given to your AWS EC2 instance.

    If successful, you should see the landing page of recommender system dashboard (image identical to that for the authentication above gif).

10. To keep your app running continuously in the background, detach from the Tmux window by pressing `ctrl + b` and then `d`. This should return you to the view of your terminal before you opened the Tmux window.

    To go back to your Tmux window at any time (even if you've left your `ssh` session and then return), simply type `tmux attach-session`.

    To see more functionality of the Tmux command, type `man tmux`.

Having run your web app within Tmux, you should be now free to end your ssh session while your webserver carries on purring along. Well done :zap:!

### Resources
You can find the detailed instructions for cloud deployment in the links below:

* [Showcase you Streamlit Web App to the world with AWS EC2](https://medium.com/swlh/showcase-you-streamlit-web-app-to-the-world-with-aws-ec2-db603c69aa28)
* [Tmux cheatsheet](https://gist.github.com/henrik/1967800)
* [Transferring Files between your laptop and Amazon instance](https://angus.readthedocs.io/en/2014/amazon/transfer-files-between-instance.html)
