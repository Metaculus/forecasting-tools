The Cursor version of devcontiners doesn't set up git automatically. To set up an SSH key with GitHub, follow these steps:


### **0. Set up git config**
Run

```
  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"
```

This may be all the is needed. If ssh isn't set up automatically by dev container, then do the below


### **1. Check for existing SSH keys**

Open a terminal and run:

```bash
ls -al ~/.ssh
```

Look for files named `id_rsa`, `id_ecdsa`, `id_ed25519` or similar (and their `.pub` counterparts).

---

### **2. Generate a new SSH key (if needed)**

If you don’t already have an SSH key or want a new one:

```bash
ssh-keygen -t ed25519
```

> If you're using an older system that doesn't support `ed25519`, use `rsa` instead:
>
> ```bash
> ssh-keygen -t rsa -b 4096
> ```

* When prompted:

  * **File to save the key**: Press Enter to accept default (`~/.ssh/id_ed25519`)
  * **Passphrase**: Optional but recommended for added security

---

### **3. Add your SSH key to the ssh-agent**

Start the SSH agent:

```bash
eval "$(ssh-agent -s)"
```

Then add your SSH key:

```bash
ssh-add ~/.ssh/id_ed25519
```

---

### **4. Add your SSH key to GitHub**

1. Copy your public SSH key to the clipboard:

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

   Then copy the output (starts with `ssh-ed25519`).

2. Go to [GitHub.com](https://github.com) and log in.

3. Click your profile picture > **Settings** > **SSH and GPG keys**

4. Click **New SSH key**

5. Paste your key, give it a title, and click **Add SSH key**

---

### **5. Test your SSH connection**

```bash
ssh -T git@github.com
```

* The first time, you’ll be asked to confirm the connection.
* You should see a message like:

```bash
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

---

Let me know if you hit any errors or want to configure it for a specific project or multiple GitHub accounts.
