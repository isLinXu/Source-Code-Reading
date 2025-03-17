source /etc/taiji/environ
echo $NODE_IP_LIST > env.txt 2>&1 
sed "s/:/ slots=/g" env.txt | sed "s/,/\n/g" >  "hostfile"
sed "s/:.//g" env.txt | sed "s/,/\n/g" >  "pssh.hosts"

pssh -i -t 0 -h pssh.hosts "pip install -r requirements.txt"