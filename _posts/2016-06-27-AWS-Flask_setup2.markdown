---
layout: post
title:  "AWS setup for flask"
date:   2016-06-23
categories: post
tags: aws flask javascript
author: Sage Elliott
published: true
---

Polaroid semiotics messenger bag whatever cronut occupy. Kinfolk crucifix austin, celiac semiotics farm-to-table hoodie typewriter single-origin coffee raw denim waistcoat banjo freegan aesthetic.

 Taxidermy master cleanse gochujang, twee yuccie ramps celiac craft beer paleo freegan iPhone jean shorts intelligentsia gluten-free raw denim. Wolf banjo squid, yuccie stumptown trust fund truffaut pitchfork. Poutine tumblr iPhone sustainable normcore helvetica. Pitchfork offal franzen iPhone slow-carb. Chia cred kickstarter drinking vinegar artisan mumblecore.



# Linux Server Configuration


### Tasks given and method for completion:

* Launch your Virtual Machine with your Udacity account
    * Must be logged into your Udacity account.
    * Visit this [link](https://www.udacity.com/account#!/development_environment) and press Create Development Environment.


* Follow the instructions provided to SSH into your server
    * Download private key
    * Move the private key file into the folder ~/.ssh (where ~ is your environment's home directory). So if you downloaded the file to the Downloads folder, just execute the following command in your terminal. `mv ~/Downloads/udacity_key.rsa ~/.ssh/`
    * Open your terminal and type in `chmod 600 ~/.ssh/udacity_key.rsa`
    * In your terminal, type in `ssh -i ~/.ssh/udacity_key.rsa root@52.27.192.5`


* Create a new user named grader
    * `sudo adduser grader`
    * optional: install finger to check user has been added `apt-get install finger`
    * `finger grader`


* Give the grader the permission to sudo
    * The README file in the /etc/sudoers.d says:"please note that using the visudo command is the recommended way to update sudoers content, since it protects against many failure modes." so that's what we will do!
    * `sudo visudo`
    * inside the file add `grader   ALL=(ALL:ALL) ALL` below the root user under "#User privilege specification"
    * save file(nano: `ctrl+x`, `Y`, Enter)


* Update all currently installed packages
    * Find updates:`sudo apt-get update`
    * Install updates:`sudo sudo apt-get upgrade` Hit Y for yes and give yourself a break while it installs.
* Change the SSH port from 22 to 2200 and other SSH configuration required from [grading rubic](https://www.udacity.com/course/viewer#!/c-nd004/l-3573679011/m-3608778867)
    * `nano /etc/ssh/sshd_config` change `port 22` to `port 2200`
    * while in the file also change `PermitRootLogin without-password` to `PermitRootLogin no` to disallow root login
    * Change `PasswordAuthentication` from `no` to `yes`. We will change back after finishing SHH login setup
    * append `AllowUsers grader ` inside file to allow grade to login through SSH
    * save file(nano: `ctrl+x`, `Y`, Enter)
    * restart ssh service`sudo service ssh reload`


* Create SSH keys and copy to server manually:
    * On your local machine generate SSH key pair with: `ssh-keygen`
    * save youkeygen file in your ssh directory `/Users/username/.ssh/` example full file path that could be used: `/Users/username/.ssh/project5`
    * You can add a password to use encase your keygen file gets compromised(you will be prompted to enter this password when you connect with key pair)
    * login into grader account using password set during user creation `ssh -v grader@*Public-IP-Address* -p 2200`
    * Make .ssh directory`mkdir .ssh`
    * make file to store key`touch .ssh/authorized_keys`
    * On your local machine read contents of the public key `cat .ssh/project5.pub`
    * Copy the key and paste in the file you just created in grader `nano
.ssh/authorized_keys` paste contents(ctr+v)
    * save file(nano: `ctrl+x`, `Y`, Enter)
    * Set permissions for files: `chmod 700 .ssh` `chmod 644 .ssh/authorized_keys`
    * Change `PasswordAuthentication` from `yes` back to `no`.  `nano /etc/ssh/sshd_config`
    * save file(nano: `ctrl+x`, `Y`, Enter)
    * login with key pair: `ssh grader@Public-IP-Address* -p 2200 -i ~/.ssh/project5`

    * alternatively you can use a shorter method found [here](https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server)


* Configure the Uncomplicated Firewall (UFW) to only allow  incoming connections for SSH (port 2200), HTTP (port 80),  and NTP (port 123)
    * Check UFW status to make sure its inactive`sudo ufw status`
    * Deny all incoming by default`sudo ufw default deny incoming`
    * Allow outgoing by default`sudo ufw default allow outgoing`
    * Allow SSH on port 2200`sudo ufw allow 2200/tcp`
    * Allow HTTP on port 80`sudo ufw allow 80/tcp`
    * Allow NTP on port 123`sudo ufw allow 123/udp`
    * Turn on firewall`sudo ufw enable`


* Configure the local timezone to UTC
    * run `sudo dpkg-reconfigure tzdata` from prompt: select none of the above. Then select UTC.


* Install and configure Apache to serve a Python mod_wsgi application
    * `sudo apt-get install apache2` Check if "It works!" at you public IP address given during setup.
    * install mod_wsgi: `sudo apt-get install libapache2-mod-wsgi`
    * configure Apache to handle requests using the WSGI module `sudo nano /etc/apache2/sites-enabled/000-default.conf`
    * add `WSGIScriptAlias / /var/www/html/myapp.wsgi` before `</   VirtualHost>` closing line
    * save file(nano: `ctrl+x`, `Y`, Enter)
    * Restart Apache `sudo apache2ctl restart`


* Install git, clone and setup your Catalog App project (from your GitHub repository from earlier in the Nanodegree program) so that it functions correctly when visiting your server’s IP address in a browser. Remember to set this up appropriately so that your .git directory is not publicly accessible via a browser!

* install git
    * `sudo apt-get install git`
    * `git config --global user.name "YOURNAME"`
    * `git config --global user.email "YOU@DOMAIN.com"`

* install python dev and verify WSGI is enabled
    * Install python-dev package`sudo apt-get install python-dev`
    * Verify wsgi is enabled `sudo a2enmod wsgi`
* Create flask app taken from [digitalocean](https://www.digitalocean.com/community/tutorials/how-to-deploy-a-flask-application-on-an-ubuntu-vps)
    * `cd /var/www`
    * `sudo mkdir catalog`
    * `cd catalog`
    * `sudo mkdir catalog`
    * `cd catalog`
    * `sudo mkdir static templates`
    * `sudo nano __init__.py `

    ```
     from flask import Flask
    app = Flask(__name__)
    @app.route("/")
    def hello():
        return "Hello, world (Testing!)"
    if __name__ == "__main__":
    app.run()
    ```

* install flask
    * `sudo apt-get install python-pip`
    * `sudo pip install virtualenv `
    * `sudo virtualenv venv`
    * `sudo chmod -R 777 venv`
    * `source venv/bin/activate`
    * `pip install Flask`
    * `python __init__.py`
    * `deactivate`

* Configure And Enable New Virtual Host
    * Create host config file `sudo nano /etc/apache2/sites-available/catalog.conf`
    * paste the following:

    ```
    <VirtualHost *:80>
      ServerName 52.27.192.5
      ServerAdmin admin@52.27.192.5
      WSGIScriptAlias / /var/www/catalog/catalog.wsgi
      <Directory /var/www/catalog/catalog/>
          Order allow,deny
          Allow from all
      </Directory>
      Alias /static /var/www/catalog/catalog/static
      <Directory /var/www/catalog/catalog/static/>
          Order allow,deny
          Allow from all
      </Directory>
      ErrorLog ${APACHE_LOG_DIR}/error.log
      LogLevel warn
      CustomLog ${APACHE_LOG_DIR}/access.log combined
  </VirtualHost>
```
    * save file(nano: `ctrl+x`, `Y`, Enter)
    * Enable `sudo a2ensite catalog`

* Create the wsgi file
    * `cd /var/www/catalog`
    * `sudo nano catalog.wsgi`

    ```
  #!/usr/bin/python
  import sys
  import logging
  logging.basicConfig(stream=sys.stderr)
  sys.path.insert(0,"/var/www/catalog/")

  from catalog import app as application
  application.secret_key = 'Add your secret key'
  ```

  * save file(nano: `ctrl+x`, `Y`, Enter)

  * `sudo service apache2 restart`

* Clone Github Repo
    * `sudo git clone https://github.com/sageio/devpost.git`
    * make sure you get hidden files iin move `shopt -s dotglob`. Move files from clone directory to catalog `mv /var/www/catalog/devpost/* /var/www/catalog/catalog/`
    * remove clone directory `sudo rm -r devpost`

* make .git inaccessible
    * from `cd /var/www/catalog/` create .htaccess file `sudo nano .htaccess`
    * paste in `RedirectMatch 404 /\.git`
    * save file(nano: `ctrl+x`, `Y`, Enter)

* install dependencies:
    * `source venv/bin/activate`
    * `pip install httplib2`
    * `pip install requests`
    * `sudo pip install --upgrade oauth2client`
    * `sudo pip install sqlalchemy`
    * `pip install Flask-SQLAlchemy`
    * `sudo pip install python-psycopg2`
    * If you used any other packages in your project be sure to install those as well.


* Install and configure PostgreSQL:
    * Install postgres`sudo apt-get install postgresql`
    * install additional models`sudo apt-get install postgresql-contrib`
    * by default no remote connections are [not allowed](http://www.postgresql.org/docs/9.2/static/auth-pg-hba-conf.html)
    * config database_setup.py `sudo nano database_setup.py`
    * `python engine = create_engine('postgresql://catalog:db-password@localhost/catalog')`
    * repeat for application.py(main.py)
    * copy your main app.py file into the __init__.py file `mv app.py __init__.py`
    * Add catalog user `sudo adduser catalog`
    * login as postgres super user`sudo su - postgres`
    * enter postgres`psql`
    * Create user catalog`CREATE USER catalog WITH PASSWORD 'db-password';`
    * Change role of user catalog to creatDB` ALTER USER catalog CREATEDB;`
    * List all users and roles to verify`\du`
    * Create new DB "catalog" with own of catalog`CREATE DATABASE catalog WITH OWNER catalog;`
    * Connect to database`\c catalog`
    * Revoke all rights `REVOKE ALL ON SCHEMA public FROM public;`
    * Give accessto only catalog role`GRANT ALL ON SCHEMA public TO catalog;`
    *   Quit postgres`\q`
    *  logout from postgres super user`exit`
    *  Setup your database schema `python database_setup.py`

    * I had problems importing psycopg2 [this](http://stackoverflow.com/questions/5629368/installing-psycopg2-into-virtualenv-when-postgresql-is-not-installed-on-developm) stack overflow post helped me
    * retstart apache `sudo service apache2 restart`

    * I was getting a `No such file or directory: 'client_secrets.json'` error. I fixed using a raw path to the file `open(r'/var/www/catalog/catalog/client_secrets.json', 'r').read())...` You'll also need to do this for any other instances of the file path
    [stack overflow](http://stackoverflow.com/questions/12201928/python-open-method-ioerror-errno-2-no-such-file-or-directory)


* fix OAuth to work with hosted Application
        * Google wont allow the IP address to make redirects so we need to set up the host name address to be usable.
    * go to [http://www.hcidata.info/host2ip.cgi](http://www.hcidata.info/host2ip.cgi) to get your host name by entering your public IP address Udacity gave you.
    * open apache configbfile `sudo nano /etc/apache2/sites-available/catalog.conf`
    * below the `ServerAdmin` paste `ServerAlias YOURHOSTNAME`
    * make sure the virtual host is enabled `sudo a2ensite catalog`
    * restart apache server `sudo service apache2 restart`
    * in your google developer console add your host name and IP address to Authorized Javascript origins. And add YOURHOSTNAME/ouath2callback to the Authorized redirect URIs.
    * Note that it may take a few minutes for you to see changes, so if you still can't login right away don't panic!


## Exceeds specs requirements


* Install glances for monitoring (Exceeds specs requirement)
    * install glances `sudo pip install Glances`
    * run `glances` to see monitor

* Configure firewall to monitor for unsuccessful attempts and use cron scripts to automatically manage packages (Exceeds specs requirement)

    * install fail2band `sudo apt-get install fail2ban`
    * copy config file to .local `sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local`
    * open .local config file to set parameters `sudo nano /etc/fail2ban/jail.local` Make at least the following changes. You can change bantime or other settings as well. This will allow email with log info and set SSH to the correct port.

    ```
  destemail = YOURNAME@DOMAIN
  action = %(action_mwl)s
  under [ssh] change port = 2220
  ```

  * install sendmail`sudo apt-get install nginx sendmail`
  * stop service `sudo service fail2ban stop`
  * Start service `sudo service fail2ban start`

  * install unattended-upgrades `sudo apt-get install unattended-upgrades`
  * enable `sudo dpkg-reconfigure -plow unattended-upgrades`
  * Automatically install security updates

## Sources used:
[Configuring Linux Web Servers - Udacity.com](https://www.udacity.com/course/viewer#!/c-ud299-nd/l-4378692847/m-4799370031)

[Get rid of sudo error message - askubuntu](http://askubuntu.com/questions/59458/error-message-when-i-run-sudo-unable-to-resolve-host-none)

[Create new ssh user - askubuntu](http://askubuntu.com/questions/16650/create-a-new-ssh-user-on-ubuntu-server)

[More info on UFW - Ubuntu docs](https://help.ubuntu.com/community/UFW)

[Change timezone - Ubuntu docs](https://help.ubuntu.com/community/UbuntuTime#Using_the_Command_Line_.28terminal.29)

[Flask App on Ubuntu VPS - DigitalOcean](https://www.digitalocean.com/community/tutorials/how-to-deploy-a-flask-application-on-an-ubuntu-vps)

[python packages not installing in virtualenv using pip - stackoverflow](http://stackoverflow.com/questions/14695278/python-packages-not-installing-in-virtualenv-using-pip)

[Move files from one directtory to another - stack exchange](http://unix.stackexchange.com/questions/6393/how-do-you-move-all-files-including-hidden-from-one-directory-to-another)

[Make .git directory web inaccessible](http://stackoverflow.com/questions/6142437/make-git-directory-web-inaccessible)

[Fix google + Oauth issue - Udacity Forum](https://discussions.udacity.com/t/google-sign-in-problems/28191)

[Using glances - Glances docs](http://glances.readthedocs.org/en/latest/glances-doc.html#configuration)

[Use fail2ban to monitor login - digital ocean](https://www.digitalocean.com/community/tutorials/how-to-protect-ssh-with-fail2ban-on-ubuntu-14-04)

[Python 2 in virtual enviroment](http://joebergantine.com/blog/2015/apr/30/installing-python-2-and-python-3-alongside-each-ot/)

[unattended upgardes - ubuntu docs](https://help.ubuntu.com/community/AutomaticSecurityUpdates)

[Using postgres in flask](http://blog.sahildiwan.com/posts/flask-and-postgresql-app-deployed-on-heroku/)
