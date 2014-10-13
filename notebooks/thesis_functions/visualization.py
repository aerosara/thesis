



<!DOCTYPE html>
<html lang="en" class="">
  <head prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# object: http://ogp.me/ns/object# article: http://ogp.me/ns/article# profile: http://ogp.me/ns/profile#">
    <meta charset='utf-8'>
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta http-equiv="Content-Language" content="en">
    
    
    <title>thesis/visualization.py at pandas · aerosara/thesis</title>
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
    <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
    <link rel="apple-touch-icon" sizes="57x57" href="/apple-touch-icon-114.png">
    <link rel="apple-touch-icon" sizes="114x114" href="/apple-touch-icon-114.png">
    <link rel="apple-touch-icon" sizes="72x72" href="/apple-touch-icon-144.png">
    <link rel="apple-touch-icon" sizes="144x144" href="/apple-touch-icon-144.png">
    <meta property="fb:app_id" content="1401488693436528">

      <meta content="@github" name="twitter:site" /><meta content="summary" name="twitter:card" /><meta content="aerosara/thesis" name="twitter:title" /><meta content="Contribute to thesis development by creating an account on GitHub." name="twitter:description" /><meta content="https://avatars3.githubusercontent.com/u/5756615?v=2&amp;s=400" name="twitter:image:src" />
<meta content="GitHub" property="og:site_name" /><meta content="object" property="og:type" /><meta content="https://avatars3.githubusercontent.com/u/5756615?v=2&amp;s=400" property="og:image" /><meta content="aerosara/thesis" property="og:title" /><meta content="https://github.com/aerosara/thesis" property="og:url" /><meta content="Contribute to thesis development by creating an account on GitHub." property="og:description" />

      <meta name="browser-stats-url" content="/_stats">
    <link rel="assets" href="https://assets-cdn.github.com/">
    <link rel="conduit-xhr" href="https://ghconduit.com:25035">
    <link rel="xhr-socket" href="/_sockets">
    <meta name="pjax-timeout" content="1000">

    <meta name="msapplication-TileImage" content="/windows-tile.png">
    <meta name="msapplication-TileColor" content="#ffffff">
    <meta name="selected-link" value="repo_source" data-pjax-transient>
      <meta name="google-analytics" content="UA-3769691-2">

    <meta content="collector.githubapp.com" name="octolytics-host" /><meta content="collector-cdn.github.com" name="octolytics-script-host" /><meta content="github" name="octolytics-app-id" /><meta content="62CC3015:4287:6329C99:543BFB1D" name="octolytics-dimension-request_id" /><meta content="5756615" name="octolytics-actor-id" /><meta content="aerosara" name="octolytics-actor-login" /><meta content="5bb0a0f731c45dc5897211c169630b5725eeee2e3e386d1324ac985033794cef" name="octolytics-actor-hash" />
    <meta content="Rails, view, blob#show" name="analytics-event" />

    
    
    <link rel="icon" type="image/x-icon" href="https://assets-cdn.github.com/favicon.ico">


    <meta content="authenticity_token" name="csrf-param" />
<meta content="Uhik5dTluYacv/1ldhkgfqyR3RjEmYJxle8o0BmQ4XGF1f+d946uIPQ86EnUurf0K4g604WhIeRi9djGYzcahQ==" name="csrf-token" />

    <link href="https://assets-cdn.github.com/assets/github-50741a8890c237738dd5333543cbc6e7336d7ac552e8e023403c24d38efbcadc.css" media="all" rel="stylesheet" type="text/css" />
    <link href="https://assets-cdn.github.com/assets/github2-f97cae5c72db1b1729daa66251ec6bbfed848d4af992c2f4842aed69d5cc5277.css" media="all" rel="stylesheet" type="text/css" />
    
    


    <meta http-equiv="x-pjax-version" content="53a3307242ee7b6c4048bb383d4b2904">

      
  <meta name="description" content="Contribute to thesis development by creating an account on GitHub.">
  <meta name="go-import" content="github.com/aerosara/thesis git https://github.com/aerosara/thesis.git">

  <meta content="5756615" name="octolytics-dimension-user_id" /><meta content="aerosara" name="octolytics-dimension-user_login" /><meta content="18041694" name="octolytics-dimension-repository_id" /><meta content="aerosara/thesis" name="octolytics-dimension-repository_nwo" /><meta content="true" name="octolytics-dimension-repository_public" /><meta content="false" name="octolytics-dimension-repository_is_fork" /><meta content="18041694" name="octolytics-dimension-repository_network_root_id" /><meta content="aerosara/thesis" name="octolytics-dimension-repository_network_root_nwo" />
  <link href="https://github.com/aerosara/thesis/commits/pandas.atom" rel="alternate" title="Recent Commits to thesis:pandas" type="application/atom+xml">

  </head>


  <body class="logged_in  env-production macintosh vis-public page-blob">
    <a href="#start-of-content" tabindex="1" class="accessibility-aid js-skip-to-content">Skip to content</a>
    <div class="wrapper">
      
      
      
      


      <div class="header header-logged-in true" role="banner">
  <div class="container clearfix">

    <a class="header-logo-invertocat" href="https://github.com/" data-hotkey="g d" aria-label="Homepage" ga-data-click="Header, go to dashboard, icon:logo">
  <span class="mega-octicon octicon-mark-github"></span>
</a>


      <div class="site-search repo-scope js-site-search" role="search">
          <form accept-charset="UTF-8" action="/aerosara/thesis/search" class="js-site-search-form" data-global-search-url="/search" data-repo-search-url="/aerosara/thesis/search" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
  <input type="text"
    class="js-site-search-field is-clearable"
    data-hotkey="s"
    name="q"
    placeholder="Search"
    data-global-scope-placeholder="Search GitHub"
    data-repo-scope-placeholder="Search"
    tabindex="1"
    autocapitalize="off">
  <div class="scope-badge">This repository</div>
</form>
      </div>
      <ul class="header-nav left" role="navigation">
        <li class="header-nav-item explore">
          <a class="header-nav-link" href="/explore" data-ga-click="Header, go to explore, text:explore">Explore</a>
        </li>
          <li class="header-nav-item">
            <a class="header-nav-link" href="https://gist.github.com" data-ga-click="Header, go to gist, text:gist">Gist</a>
          </li>
          <li class="header-nav-item">
            <a class="header-nav-link" href="/blog" data-ga-click="Header, go to blog, text:blog">Blog</a>
          </li>
        <li class="header-nav-item">
          <a class="header-nav-link" href="https://help.github.com" data-ga-click="Header, go to help, text:help">Help</a>
        </li>
      </ul>

    
<ul class="header-nav user-nav right" id="user-links">
  <li class="header-nav-item dropdown js-menu-container">
    <a class="header-nav-link name" href="/aerosara" data-ga-click="Header, go to profile, text:username">
      <img alt="Sara Case" class="avatar" data-user="5756615" height="20" src="https://avatars2.githubusercontent.com/u/5756615?v=2&amp;s=40" width="20" />
      <span class="css-truncate">
        <span class="css-truncate-target">aerosara</span>
      </span>
    </a>
  </li>

  <li class="header-nav-item dropdown js-menu-container">
    <a class="header-nav-link js-menu-target tooltipped tooltipped-s" href="#" aria-label="Create new..." data-ga-click="Header, create new, icon:add">
      <span class="octicon octicon-plus"></span>
      <span class="dropdown-caret"></span>
    </a>

    <div class="dropdown-menu-content js-menu-content">
      
<ul class="dropdown-menu">
  <li>
    <a href="/new"><span class="octicon octicon-repo"></span> New repository</a>
  </li>
  <li>
    <a href="/organizations/new"><span class="octicon octicon-organization"></span> New organization</a>
  </li>


    <li class="dropdown-divider"></li>
    <li class="dropdown-header">
      <span title="aerosara/thesis">This repository</span>
    </li>
      <li>
        <a href="/aerosara/thesis/issues/new"><span class="octicon octicon-issue-opened"></span> New issue</a>
      </li>
      <li>
        <a href="/aerosara/thesis/settings/collaboration"><span class="octicon octicon-person"></span> New collaborator</a>
      </li>
</ul>

    </div>
  </li>

  <li class="header-nav-item">
        <a href="/notifications" aria-label="You have no unread notifications" class="header-nav-link notification-indicator tooltipped tooltipped-s" data-ga-click="Header, go to notifications, icon:read" data-hotkey="g n">
        <span class="mail-status all-read"></span>
        <span class="octicon octicon-inbox"></span>
</a>
  </li>

  <li class="header-nav-item">
    <a class="header-nav-link tooltipped tooltipped-s" href="/settings/profile" id="account_settings" aria-label="Settings" data-ga-click="Header, go to settings, icon:settings">
      <span class="octicon octicon-gear"></span>
    </a>
  </li>

  <li class="header-nav-item">
    <form accept-charset="UTF-8" action="/logout" class="logout-form" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="yPWuGW147Wh8PpZLGiwVyHwNlgBUUN/dMjENeJlvpgP1OoWjSwgnzT24HY/bsp81p9gOauvgkTbCGloE2q0PfA==" /></div>
      <button class="header-nav-link sign-out-button tooltipped tooltipped-s" aria-label="Sign out" data-ga-click="Header, sign out, icon:logout">
        <span class="octicon octicon-sign-out"></span>
      </button>
</form>  </li>

</ul>


    
  </div>
</div>

      

        


      <div id="start-of-content" class="accessibility-aid"></div>
          <div class="site" itemscope itemtype="http://schema.org/WebPage">
    <div id="js-flash-container">
      
    </div>
    <div class="pagehead repohead instapaper_ignore readability-menu">
      <div class="container">
        
<ul class="pagehead-actions">

    <li class="subscription">
      <form accept-charset="UTF-8" action="/notifications/subscribe" class="js-social-container" data-autosubmit="true" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="VY0caeGBROEl9D+j/NKU12Goywe1Hzyq3YeCWYHT6aQ47OVy5lZfZMaA71wSaObilnwG8Xyf1jbMbV6d4sd+QA==" /></div>  <input id="repository_id" name="repository_id" type="hidden" value="18041694" />

    <div class="select-menu js-menu-container js-select-menu">
      <a class="social-count js-social-count" href="/aerosara/thesis/watchers">
        2
      </a>
      <a href="/aerosara/thesis/subscription"
        class="minibutton select-menu-button with-count js-menu-target" role="button" tabindex="0" aria-haspopup="true">
        <span class="js-select-button">
          <span class="octicon octicon-eye"></span>
          Unwatch
        </span>
      </a>

      <div class="select-menu-modal-holder">
        <div class="select-menu-modal subscription-menu-modal js-menu-content" aria-hidden="true">
          <div class="select-menu-header">
            <span class="select-menu-title">Notifications</span>
            <span class="octicon octicon-x js-menu-close" role="button" aria-label="Close"></span>
          </div> <!-- /.select-menu-header -->

          <div class="select-menu-list js-navigation-container" role="menu">

            <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input id="do_included" name="do" type="radio" value="included" />
                <h4>Not watching</h4>
                <span class="description">Be notified when participating or @mentioned.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-eye"></span>
                  Watch
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

            <div class="select-menu-item js-navigation-item selected" role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input checked="checked" id="do_subscribed" name="do" type="radio" value="subscribed" />
                <h4>Watching</h4>
                <span class="description">Be notified of all conversations.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-eye"></span>
                  Unwatch
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

            <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input id="do_ignore" name="do" type="radio" value="ignore" />
                <h4>Ignoring</h4>
                <span class="description">Never be notified.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-mute"></span>
                  Stop ignoring
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

          </div> <!-- /.select-menu-list -->

        </div> <!-- /.select-menu-modal -->
      </div> <!-- /.select-menu-modal-holder -->
    </div> <!-- /.select-menu -->

</form>
    </li>

  <li>
    
  <div class="js-toggler-container js-social-container starring-container ">

    <form accept-charset="UTF-8" action="/aerosara/thesis/unstar" class="js-toggler-form starred js-unstar-button" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="aCPCw8aSI6HhREbTF09m5c+toZcmvSX9INdlZMfQC7kjy9p3Z+ESXSKiJsTTFBETZL6UgOFNVnh2x8t8ov2znA==" /></div>
      <button
        class="minibutton with-count js-toggler-target star-button"
        aria-label="Unstar this repository" title="Unstar aerosara/thesis">
        <span class="octicon octicon-star"></span>
        Unstar
      </button>
        <a class="social-count js-social-count" href="/aerosara/thesis/stargazers">
          0
        </a>
</form>
    <form accept-charset="UTF-8" action="/aerosara/thesis/star" class="js-toggler-form unstarred js-star-button" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="aATH1tACobxai2TihJK02MnGJ3U6dZzkIlsRPTTC+yBb+oKKr2QCnqKjwpo1D0ZjlnC7uqlkep1alKWnQ912Hw==" /></div>
      <button
        class="minibutton with-count js-toggler-target star-button"
        aria-label="Star this repository" title="Star aerosara/thesis">
        <span class="octicon octicon-star"></span>
        Star
      </button>
        <a class="social-count js-social-count" href="/aerosara/thesis/stargazers">
          0
        </a>
</form>  </div>

  </li>


        <li>
          <a href="/aerosara/thesis/fork" class="minibutton with-count js-toggler-target fork-button tooltipped-n" title="Fork your own copy of aerosara/thesis to your account" aria-label="Fork your own copy of aerosara/thesis to your account" rel="nofollow" data-method="post">
            <span class="octicon octicon-repo-forked"></span>
            Fork
          </a>
          <a href="/aerosara/thesis/network" class="social-count">0</a>
        </li>

</ul>

        <h1 itemscope itemtype="http://data-vocabulary.org/Breadcrumb" class="entry-title public">
          <span class="mega-octicon octicon-repo"></span>
          <span class="author"><a href="/aerosara" class="url fn" itemprop="url" rel="author"><span itemprop="title">aerosara</span></a></span><!--
       --><span class="path-divider">/</span><!--
       --><strong><a href="/aerosara/thesis" class="js-current-repository js-repo-home-link">thesis</a></strong>

          <span class="page-context-loader">
            <img alt="" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
          </span>

        </h1>
      </div><!-- /.container -->
    </div><!-- /.repohead -->

    <div class="container">
      <div class="repository-with-sidebar repo-container new-discussion-timeline  ">
        <div class="repository-sidebar clearfix">
            
<div class="sunken-menu vertical-right repo-nav js-repo-nav js-repository-container-pjax js-octicon-loaders" role="navigation" data-issue-count-url="/aerosara/thesis/issues/counts">
  <div class="sunken-menu-contents">
    <ul class="sunken-menu-group">
      <li class="tooltipped tooltipped-w" aria-label="Code">
        <a href="/aerosara/thesis/tree/pandas" aria-label="Code" class="selected js-selected-navigation-item sunken-menu-item" data-hotkey="g c" data-pjax="true" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches /aerosara/thesis/tree/pandas">
          <span class="octicon octicon-code"></span> <span class="full-word">Code</span>
          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>

        <li class="tooltipped tooltipped-w" aria-label="Issues">
          <a href="/aerosara/thesis/issues" aria-label="Issues" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-hotkey="g i" data-selected-links="repo_issues repo_labels repo_milestones /aerosara/thesis/issues">
            <span class="octicon octicon-issue-opened"></span> <span class="full-word">Issues</span>
            <span class="js-issue-replace-counter"></span>
            <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>        </li>

      <li class="tooltipped tooltipped-w" aria-label="Pull Requests">
        <a href="/aerosara/thesis/pulls" aria-label="Pull Requests" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-hotkey="g p" data-selected-links="repo_pulls /aerosara/thesis/pulls">
            <span class="octicon octicon-git-pull-request"></span> <span class="full-word">Pull Requests</span>
            <span class="js-pull-replace-counter"></span>
            <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>


        <li class="tooltipped tooltipped-w" aria-label="Wiki">
          <a href="/aerosara/thesis/wiki" aria-label="Wiki" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-hotkey="g w" data-selected-links="repo_wiki /aerosara/thesis/wiki">
            <span class="octicon octicon-book"></span> <span class="full-word">Wiki</span>
            <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>        </li>
    </ul>
    <div class="sunken-menu-separator"></div>
    <ul class="sunken-menu-group">

      <li class="tooltipped tooltipped-w" aria-label="Pulse">
        <a href="/aerosara/thesis/pulse/weekly" aria-label="Pulse" class="js-selected-navigation-item sunken-menu-item" data-pjax="true" data-selected-links="pulse /aerosara/thesis/pulse/weekly">
          <span class="octicon octicon-pulse"></span> <span class="full-word">Pulse</span>
          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>

      <li class="tooltipped tooltipped-w" aria-label="Graphs">
        <a href="/aerosara/thesis/graphs" aria-label="Graphs" class="js-selected-navigation-item sunken-menu-item" data-pjax="true" data-selected-links="repo_graphs repo_contributors /aerosara/thesis/graphs">
          <span class="octicon octicon-graph"></span> <span class="full-word">Graphs</span>
          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>
    </ul>


      <div class="sunken-menu-separator"></div>
      <ul class="sunken-menu-group">
        <li class="tooltipped tooltipped-w" aria-label="Settings">
          <a href="/aerosara/thesis/settings" aria-label="Settings" class="js-selected-navigation-item sunken-menu-item" data-pjax="true" data-selected-links="repo_settings /aerosara/thesis/settings">
            <span class="octicon octicon-tools"></span> <span class="full-word">Settings</span>
            <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>        </li>
      </ul>
  </div>
</div>

              <div class="only-with-full-nav">
                
  
<div class="clone-url open"
  data-protocol-type="http"
  data-url="/users/set_protocol?protocol_selector=http&amp;protocol_type=push">
  <h3><span class="text-emphasized">HTTPS</span> clone URL</h3>
  <div class="input-group">
    <input type="text" class="input-mini input-monospace js-url-field"
           value="https://github.com/aerosara/thesis.git" readonly="readonly">
    <span class="input-group-button">
      <button aria-label="Copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="https://github.com/aerosara/thesis.git" data-copied-hint="Copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>

  
<div class="clone-url "
  data-protocol-type="ssh"
  data-url="/users/set_protocol?protocol_selector=ssh&amp;protocol_type=push">
  <h3><span class="text-emphasized">SSH</span> clone URL</h3>
  <div class="input-group">
    <input type="text" class="input-mini input-monospace js-url-field"
           value="git@github.com:aerosara/thesis.git" readonly="readonly">
    <span class="input-group-button">
      <button aria-label="Copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="git@github.com:aerosara/thesis.git" data-copied-hint="Copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>

  
<div class="clone-url "
  data-protocol-type="subversion"
  data-url="/users/set_protocol?protocol_selector=subversion&amp;protocol_type=push">
  <h3><span class="text-emphasized">Subversion</span> checkout URL</h3>
  <div class="input-group">
    <input type="text" class="input-mini input-monospace js-url-field"
           value="https://github.com/aerosara/thesis" readonly="readonly">
    <span class="input-group-button">
      <button aria-label="Copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="https://github.com/aerosara/thesis" data-copied-hint="Copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>


<p class="clone-options">You can clone with
      <a href="#" class="js-clone-selector" data-protocol="http">HTTPS</a>,
      <a href="#" class="js-clone-selector" data-protocol="ssh">SSH</a>,
      or <a href="#" class="js-clone-selector" data-protocol="subversion">Subversion</a>.
  <a href="https://help.github.com/articles/which-remote-url-should-i-use" class="help tooltipped tooltipped-n" aria-label="Get help on which URL is right for you.">
    <span class="octicon octicon-question"></span>
  </a>
</p>

  <a href="http://mac.github.com" data-url="github-mac://openRepo/https://github.com/aerosara/thesis" class="minibutton sidebar-button js-conduit-rewrite-url" title="Save aerosara/thesis to your computer and use it in GitHub Desktop." aria-label="Save aerosara/thesis to your computer and use it in GitHub Desktop.">
    <span class="octicon octicon-device-desktop"></span>
    Clone in Desktop
  </a>


                <a href="/aerosara/thesis/archive/pandas.zip"
                   class="minibutton sidebar-button"
                   aria-label="Download the contents of aerosara/thesis as a zip file"
                   title="Download the contents of aerosara/thesis as a zip file"
                   rel="nofollow">
                  <span class="octicon octicon-cloud-download"></span>
                  Download ZIP
                </a>
              </div>
        </div><!-- /.repository-sidebar -->

        <div id="js-repo-pjax-container" class="repository-content context-loader-container" data-pjax-container>
          

<a href="/aerosara/thesis/blob/db643b11ec59cd0d316352fae33e617a96e788a3/notebooks/thesis_functions/visualization.py" class="hidden js-permalink-shortcut" data-hotkey="y">Permalink</a>

<!-- blob contrib key: blob_contributors:v21:03359560030746ca7704f21a89335bf4 -->

<div class="file-navigation">
  
<div class="select-menu js-menu-container js-select-menu left">
  <span class="minibutton select-menu-button js-menu-target css-truncate" data-hotkey="w"
    data-master-branch="master"
    data-ref="pandas"
    title="pandas"
    role="button" aria-label="Switch branches or tags" tabindex="0" aria-haspopup="true">
    <span class="octicon octicon-git-branch"></span>
    <i>branch:</i>
    <span class="js-select-button css-truncate-target">pandas</span>
  </span>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax aria-hidden="true">

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <span class="select-menu-title">Switch branches/tags</span>
        <span class="octicon octicon-x js-menu-close" role="button" aria-label="Close"></span>
      </div> <!-- /.select-menu-header -->

      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Find or create a branch…" id="context-commitish-filter-field" class="js-filterable-field js-navigation-enable" placeholder="Find or create a branch…">
        </div>
        <div class="select-menu-tabs">
          <ul>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="branches" class="js-select-menu-tab">Branches</a>
            </li>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="tags" class="js-select-menu-tab">Tags</a>
            </li>
          </ul>
        </div><!-- /.select-menu-tabs -->
      </div><!-- /.select-menu-filters -->

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="branches">

        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/aerosara/thesis/blob/master/notebooks/thesis_functions/visualization.py"
                 data-name="master"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="master">master</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item selected">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/aerosara/thesis/blob/pandas/notebooks/thesis_functions/visualization.py"
                 data-name="pandas"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="pandas">pandas</a>
            </div> <!-- /.select-menu-item -->
        </div>

          <form accept-charset="UTF-8" action="/aerosara/thesis/branches" class="js-create-branch select-menu-item select-menu-new-item-form js-navigation-item js-new-item-form" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="C7UiTatYr0sLpXML0Dm6xld8yutCJr9s3DK9HqVzQ5YHoKtgp31WnKa86BRwqm3tGBWzYSj0Rjkt8U3vsJWPDQ==" /></div>
            <span class="octicon octicon-git-branch select-menu-item-icon"></span>
            <div class="select-menu-item-text">
              <h4>Create branch: <span class="js-new-item-name"></span></h4>
              <span class="description">from ‘pandas’</span>
            </div>
            <input type="hidden" name="name" id="name" class="js-new-item-value">
            <input type="hidden" name="branch" id="branch" value="pandas">
            <input type="hidden" name="path" id="path" value="notebooks/thesis_functions/visualization.py">
          </form> <!-- /.select-menu-item -->

      </div> <!-- /.select-menu-list -->

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="tags">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div> <!-- /.select-menu-list -->

    </div> <!-- /.select-menu-modal -->
  </div> <!-- /.select-menu-modal-holder -->
</div> <!-- /.select-menu -->

  <div class="button-group right">
    <a href="/aerosara/thesis/find/pandas"
          class="js-show-file-finder minibutton empty-icon tooltipped tooltipped-s"
          data-pjax
          data-hotkey="t"
          aria-label="Quickly jump between files">
      <span class="octicon octicon-list-unordered"></span>
    </a>
    <button class="js-zeroclipboard minibutton zeroclipboard-button"
          data-clipboard-text="notebooks/thesis_functions/visualization.py"
          aria-label="Copy to clipboard"
          data-copied-hint="Copied!">
      <span class="octicon octicon-clippy"></span>
    </button>
  </div>

  <div class="breadcrumb">
    <span class='repo-root js-repo-root'><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/aerosara/thesis/tree/pandas" class="" data-branch="pandas" data-direction="back" data-pjax="true" itemscope="url"><span itemprop="title">thesis</span></a></span></span><span class="separator"> / </span><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/aerosara/thesis/tree/pandas/notebooks" class="" data-branch="pandas" data-direction="back" data-pjax="true" itemscope="url"><span itemprop="title">notebooks</span></a></span><span class="separator"> / </span><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/aerosara/thesis/tree/pandas/notebooks/thesis_functions" class="" data-branch="pandas" data-direction="back" data-pjax="true" itemscope="url"><span itemprop="title">thesis_functions</span></a></span><span class="separator"> / </span><strong class="final-path">visualization.py</strong>
  </div>
</div>


  <div class="commit commit-loader file-history-tease js-deferred-content" data-url="/aerosara/thesis/contributors/pandas/notebooks/thesis_functions/visualization.py">
    <div class="file-history-tease-header">
      Fetching contributors&hellip;
    </div>

    <div class="participation">
      <p class="loader-loading"><img alt="" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32-EAF2F5.gif" width="16" /></p>
      <p class="loader-error">Cannot retrieve contributors at this time</p>
    </div>
  </div>

<div class="file-box">
  <div class="file">
    <div class="meta clearfix">
      <div class="info file-name">
          <span>78 lines (59 sloc)</span>
          <span class="meta-divider"></span>
        <span>2.68 kb</span>
      </div>
      <div class="actions">
        <div class="button-group">
          <a href="/aerosara/thesis/raw/pandas/notebooks/thesis_functions/visualization.py" class="minibutton " id="raw-url">Raw</a>
            <a href="/aerosara/thesis/blame/pandas/notebooks/thesis_functions/visualization.py" class="minibutton js-update-url-with-hash">Blame</a>
          <a href="/aerosara/thesis/commits/pandas/notebooks/thesis_functions/visualization.py" class="minibutton " rel="nofollow">History</a>
        </div><!-- /.button-group -->

          <a class="octicon-button tooltipped tooltipped-nw js-conduit-openfile-check"
             href="http://mac.github.com"
             data-url="github-mac://openRepo/https://github.com/aerosara/thesis?branch=pandas&amp;filepath=notebooks%2Fthesis_functions%2Fvisualization.py"
             aria-label="Open this file in GitHub for Mac"
             data-failed-title="Your version of GitHub for Mac is too old to open this file. Try checking for updates.">
              <span class="octicon octicon-device-desktop"></span>
          </a>

              <a class="octicon-button js-update-url-with-hash"
                 href="/aerosara/thesis/edit/pandas/notebooks/thesis_functions/visualization.py"
                 data-method="post" rel="nofollow" data-hotkey="e"><span class="octicon octicon-pencil"></span></a>

            <a class="octicon-button danger"
               href="/aerosara/thesis/delete/pandas/notebooks/thesis_functions/visualization.py"
               data-method="post" data-test-id="delete-blob-file" rel="nofollow">
          <span class="octicon octicon-trashcan"></span>
        </a>
      </div><!-- /.actions -->
    </div>
    
  <div class="blob-wrapper data type-python">
      <table class="highlight tab-size-8 js-file-line-container">
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code js-file-line"><span class="c"># -*- coding: utf-8 -*-</span></td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code js-file-line"><span class="c"># &lt;nbformat&gt;3.0&lt;/nbformat&gt;</span></td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code js-file-line">
</td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code js-file-line"><span class="c"># &lt;codecell&gt;</span></td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code js-file-line">
</td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code js-file-line">
</td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code js-file-line"><span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="kn">as</span> <span class="nn">plt</span></td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code js-file-line"><span class="kn">from</span> <span class="nn">mpl_toolkits.mplot3d</span> <span class="kn">import</span> <span class="n">Axes3D</span></td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code js-file-line">
</td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code js-file-line"><span class="k">def</span> <span class="nf">CreatePlotGrid</span><span class="p">(</span><span class="n">title</span><span class="p">,</span> <span class="n">xlabel</span><span class="p">,</span> <span class="n">ylabel</span><span class="p">,</span> <span class="n">zlabel</span><span class="p">,</span> <span class="n">aspectmode</span><span class="p">):</span></td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code js-file-line">    </td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code js-file-line">    <span class="c"># plot</span></td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code js-file-line">    <span class="n">fig</span><span class="p">,</span> <span class="p">((</span><span class="n">axXZ</span><span class="p">,</span> <span class="n">axYZ</span><span class="p">),</span> <span class="p">(</span><span class="n">axXY</span><span class="p">,</span> <span class="n">ax3D</span><span class="p">))</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code js-file-line">    <span class="n">fig</span><span class="o">.</span><span class="n">suptitle</span><span class="p">(</span><span class="n">title</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">14</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code js-file-line">    <span class="n">fig</span><span class="o">.</span><span class="n">set_size_inches</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">10</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code js-file-line">    <span class="n">fig</span><span class="o">.</span><span class="n">subplots_adjust</span><span class="p">(</span><span class="n">hspace</span><span class="o">=</span><span class="mf">0.2</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code js-file-line">    <span class="n">fig</span><span class="o">.</span><span class="n">subplots_adjust</span><span class="p">(</span><span class="n">wspace</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code js-file-line">
</td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code js-file-line">    <span class="c"># XZ Plane</span></td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code js-file-line">    <span class="n">axXZ</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">xlabel</span> <span class="o">+</span> <span class="n">zlabel</span> <span class="o">+</span> <span class="s">&#39; Plane&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code js-file-line">    <span class="n">axXZ</span><span class="o">.</span><span class="n">xaxis</span><span class="o">.</span><span class="n">set_label_text</span><span class="p">(</span><span class="n">xlabel</span> <span class="o">+</span> <span class="s">&#39; axis&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code js-file-line">    <span class="n">axXZ</span><span class="o">.</span><span class="n">yaxis</span><span class="o">.</span><span class="n">set_label_text</span><span class="p">(</span><span class="n">zlabel</span> <span class="o">+</span> <span class="s">&#39; axis&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code js-file-line">    <span class="n">axXZ</span><span class="o">.</span><span class="n">set_aspect</span><span class="p">(</span><span class="n">aspectmode</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code js-file-line">
</td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code js-file-line">    <span class="c"># YZ Plane</span></td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code js-file-line">    <span class="n">axYZ</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">ylabel</span> <span class="o">+</span> <span class="n">zlabel</span> <span class="o">+</span> <span class="s">&#39; Plane&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code js-file-line">    <span class="n">axYZ</span><span class="o">.</span><span class="n">xaxis</span><span class="o">.</span><span class="n">set_label_text</span><span class="p">(</span><span class="n">ylabel</span> <span class="o">+</span> <span class="s">&#39; axis&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code js-file-line">    <span class="n">axYZ</span><span class="o">.</span><span class="n">yaxis</span><span class="o">.</span><span class="n">set_label_text</span><span class="p">(</span><span class="n">zlabel</span> <span class="o">+</span> <span class="s">&#39; axis&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code js-file-line">    <span class="n">axYZ</span><span class="o">.</span><span class="n">set_aspect</span><span class="p">(</span><span class="n">aspectmode</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code js-file-line">
</td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code js-file-line">    <span class="c"># XY Plane</span></td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code js-file-line">    <span class="n">axXY</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">xlabel</span> <span class="o">+</span> <span class="n">ylabel</span> <span class="o">+</span> <span class="s">&#39; Plane&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code js-file-line">    <span class="n">axXY</span><span class="o">.</span><span class="n">xaxis</span><span class="o">.</span><span class="n">set_label_text</span><span class="p">(</span><span class="n">xlabel</span> <span class="o">+</span> <span class="s">&#39; axis&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code js-file-line">    <span class="n">axXY</span><span class="o">.</span><span class="n">yaxis</span><span class="o">.</span><span class="n">set_label_text</span><span class="p">(</span><span class="n">ylabel</span> <span class="o">+</span> <span class="s">&#39; axis&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code js-file-line">    <span class="n">axXY</span><span class="o">.</span><span class="n">set_aspect</span><span class="p">(</span><span class="n">aspectmode</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code js-file-line">
</td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code js-file-line">    <span class="c"># plot in 3D</span></td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code js-file-line">    <span class="n">ax3D</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s">&#39;off&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code js-file-line">    <span class="n">ax3D</span> <span class="o">=</span> <span class="n">fig</span><span class="o">.</span><span class="n">add_subplot</span><span class="p">(</span><span class="mi">224</span><span class="p">,</span> <span class="n">projection</span><span class="o">=</span><span class="s">&#39;3d&#39;</span><span class="p">)</span> <span class="c"># &quot;224&quot; means &quot;2x2 grid, 4th subplot&quot;</span></td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code js-file-line">    <span class="n">ax3D</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s">&#39;3D View in &#39;</span> <span class="o">+</span> <span class="n">xlabel</span> <span class="o">+</span> <span class="n">ylabel</span> <span class="o">+</span> <span class="n">zlabel</span> <span class="o">+</span> <span class="s">&#39; Frame&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code js-file-line">    <span class="n">ax3D</span><span class="o">.</span><span class="n">xaxis</span><span class="o">.</span><span class="n">set_label_text</span><span class="p">(</span><span class="n">xlabel</span> <span class="o">+</span> <span class="s">&#39; axis&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code js-file-line">    <span class="n">ax3D</span><span class="o">.</span><span class="n">yaxis</span><span class="o">.</span><span class="n">set_label_text</span><span class="p">(</span><span class="n">ylabel</span> <span class="o">+</span> <span class="s">&#39; axis&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code js-file-line">    <span class="n">ax3D</span><span class="o">.</span><span class="n">zaxis</span><span class="o">.</span><span class="n">set_label_text</span><span class="p">(</span><span class="n">zlabel</span> <span class="o">+</span> <span class="s">&#39; axis&#39;</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code js-file-line">    <span class="n">ax3D</span><span class="o">.</span><span class="n">set_aspect</span><span class="p">(</span><span class="n">aspectmode</span><span class="p">)</span></td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code js-file-line">    </td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code js-file-line">    <span class="k">return</span> <span class="n">axXZ</span><span class="p">,</span> <span class="n">axYZ</span><span class="p">,</span> <span class="n">axXY</span><span class="p">,</span> <span class="n">ax3D</span></td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code js-file-line">        </td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code js-file-line">     </td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code js-file-line"><span class="c"># Allowed colors:</span></td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code js-file-line"><span class="c"># b: blue</span></td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code js-file-line"><span class="c"># g: green</span></td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code js-file-line"><span class="c"># r: red</span></td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code js-file-line"><span class="c"># c: cyan</span></td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code js-file-line"><span class="c"># m: magenta</span></td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code js-file-line"><span class="c"># y: yellow</span></td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code js-file-line"><span class="c"># k: black</span></td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code js-file-line"><span class="c"># w: white</span></td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code js-file-line">    </td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code js-file-line"><span class="k">def</span> <span class="nf">SetPlotGridData</span><span class="p">(</span><span class="n">axXZ</span><span class="p">,</span> <span class="n">axYZ</span><span class="p">,</span> <span class="n">axXY</span><span class="p">,</span> <span class="n">ax3D</span><span class="p">,</span> <span class="n">data</span><span class="p">,</span> <span class="n">points</span><span class="p">):</span></td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code js-file-line">    </td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code js-file-line">    <span class="c"># add points to plots</span></td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code js-file-line">    <span class="k">for</span> <span class="n">key</span> <span class="ow">in</span> <span class="n">points</span><span class="p">:</span></td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code js-file-line">        <span class="n">axXZ</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;xyz&#39;</span><span class="p">][</span><span class="mi">0</span><span class="p">],</span> <span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;xyz&#39;</span><span class="p">][</span><span class="mi">2</span><span class="p">],</span> <span class="s">&#39;o&#39;</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">key</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;color&#39;</span><span class="p">])</span></td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code js-file-line">        <span class="n">axYZ</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;xyz&#39;</span><span class="p">][</span><span class="mi">1</span><span class="p">],</span> <span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;xyz&#39;</span><span class="p">][</span><span class="mi">2</span><span class="p">],</span> <span class="s">&#39;o&#39;</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">key</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;color&#39;</span><span class="p">])</span></td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code js-file-line">        <span class="n">axXY</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;xyz&#39;</span><span class="p">][</span><span class="mi">0</span><span class="p">],</span> <span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;xyz&#39;</span><span class="p">][</span><span class="mi">1</span><span class="p">],</span> <span class="s">&#39;o&#39;</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">key</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;color&#39;</span><span class="p">])</span></td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code js-file-line">        <span class="n">ax3D</span><span class="o">.</span><span class="n">plot</span><span class="p">([</span><span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;xyz&#39;</span><span class="p">][</span><span class="mi">0</span><span class="p">]],</span> <span class="p">[</span><span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;xyz&#39;</span><span class="p">][</span><span class="mi">1</span><span class="p">]],</span> <span class="p">[</span><span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;xyz&#39;</span><span class="p">][</span><span class="mi">2</span><span class="p">]],</span> <span class="s">&#39;o&#39;</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">key</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">points</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;color&#39;</span><span class="p">])</span></td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code js-file-line">        </td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code js-file-line">    <span class="c"># add data to plots</span></td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code js-file-line">    <span class="k">for</span> <span class="n">key</span> <span class="ow">in</span> <span class="n">data</span><span class="p">:</span></td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code js-file-line">        <span class="n">axXZ</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;x&#39;</span><span class="p">],</span> <span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;z&#39;</span><span class="p">],</span> <span class="s">&#39;-&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">key</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;color&#39;</span><span class="p">])</span></td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code js-file-line">        <span class="n">axYZ</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;y&#39;</span><span class="p">],</span> <span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;z&#39;</span><span class="p">],</span> <span class="s">&#39;-&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">key</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;color&#39;</span><span class="p">])</span></td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code js-file-line">        <span class="n">axXY</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;x&#39;</span><span class="p">],</span> <span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;y&#39;</span><span class="p">],</span> <span class="s">&#39;-&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">key</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;color&#39;</span><span class="p">])</span></td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code js-file-line">        <span class="n">ax3D</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;x&#39;</span><span class="p">],</span> <span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;y&#39;</span><span class="p">],</span> <span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;z&#39;</span><span class="p">],</span> <span class="s">&#39;-&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">key</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">data</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s">&#39;color&#39;</span><span class="p">])</span></td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code js-file-line">        </td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code js-file-line">    <span class="c">#ax3D.legend(loc=&#39;center left&#39;, bbox_to_anchor=(1.2, 0.5))</span></td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code js-file-line">        </td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code js-file-line">
</td>
      </tr>
</table>

  </div>

  </div>
</div>

<a href="#jump-to-line" rel="facebox[.linejump]" data-hotkey="l" style="display:none">Jump to Line</a>
<div id="jump-to-line" style="display:none">
  <form accept-charset="UTF-8" class="js-jump-to-line-form">
    <input class="linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" autofocus>
    <button type="submit" class="button">Go</button>
  </form>
</div>

        </div>

      </div><!-- /.repo-container -->
      <div class="modal-backdrop"></div>
    </div><!-- /.container -->
  </div><!-- /.site -->


    </div><!-- /.wrapper -->

      <div class="container">
  <div class="site-footer" role="contentinfo">
    <ul class="site-footer-links right">
      <li><a href="https://status.github.com/">Status</a></li>
      <li><a href="http://developer.github.com">API</a></li>
      <li><a href="http://training.github.com">Training</a></li>
      <li><a href="http://shop.github.com">Shop</a></li>
      <li><a href="/blog">Blog</a></li>
      <li><a href="/about">About</a></li>

    </ul>

    <a href="/" aria-label="Homepage">
      <span class="mega-octicon octicon-mark-github" title="GitHub"></span>
    </a>

    <ul class="site-footer-links">
      <li>&copy; 2014 <span title="0.06835s from github-fe140-cp1-prd.iad.github.net">GitHub</span>, Inc.</li>
        <li><a href="/site/terms">Terms</a></li>
        <li><a href="/site/privacy">Privacy</a></li>
        <li><a href="/security">Security</a></li>
        <li><a href="/contact">Contact</a></li>
    </ul>
  </div><!-- /.site-footer -->
</div><!-- /.container -->


    <div class="fullscreen-overlay js-fullscreen-overlay" id="fullscreen_overlay">
  <div class="fullscreen-container js-suggester-container">
    <div class="textarea-wrap">
      <textarea name="fullscreen-contents" id="fullscreen-contents" class="fullscreen-contents js-fullscreen-contents js-suggester-field" placeholder=""></textarea>
    </div>
  </div>
  <div class="fullscreen-sidebar">
    <a href="#" class="exit-fullscreen js-exit-fullscreen tooltipped tooltipped-w" aria-label="Exit Zen Mode">
      <span class="mega-octicon octicon-screen-normal"></span>
    </a>
    <a href="#" class="theme-switcher js-theme-switcher tooltipped tooltipped-w"
      aria-label="Switch themes">
      <span class="octicon octicon-color-mode"></span>
    </a>
  </div>
</div>



    <div id="ajax-error-message" class="flash flash-error">
      <span class="octicon octicon-alert"></span>
      <a href="#" class="octicon octicon-x flash-close js-ajax-error-dismiss" aria-label="Dismiss error"></a>
      Something went wrong with that request. Please try again.
    </div>


      <script crossorigin="anonymous" src="https://assets-cdn.github.com/assets/frameworks-f4749195ce218608caf72b3ddefff5f580445386f2529c60e027cd18d1db0cb5.js" type="text/javascript"></script>
      <script async="async" crossorigin="anonymous" src="https://assets-cdn.github.com/assets/github-56901a1e60b184d90b134d3f30a8700dee7e5d313a3a70e28f6adb239d7d8797.js" type="text/javascript"></script>
      
      
        <script async src="https://www.google-analytics.com/analytics.js"></script>
  </body>
</html>

