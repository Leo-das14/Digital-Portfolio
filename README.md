
DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" type="image/x-icon" href="./favicon.ico">
    <link rel="stylesheet" href="./style.css">
    <title>My Digital Portfolio</title>
</head>
<body>
       
<header>
    <section>   
    <h1>ABISHEK S</h1>
        <h3>Student</h3>
        <h4>Government Arts and Science College,valparai</h4>
    </section>
    <div>
        <img src="./profilepic.jpg" alt="My Profile Photo" class="propic"/>
    </div>
</header>

<nav>
    <ul>
        <li><a href="#about">About Me</a></li>
        <li><a href="#edu">Education Qualification</a> </li>
        <li><a href="#exp">Experience</a></li>
        <li><a href="#interest">Area of Interest</a></li>
        <li> <a href="#contact">Contact</a></li>
        <li><a href="#download">Download CV</a></li>
    </ul>
</nav>

<section id="about">
    <div class="content">
        <br>
    <h2>About Me
    </h2>
    <ul>
        <li>I'm ABISHEK S i am studing in Governemnt Arts and Science College,Valparai</li>
       
    </ul>
</div>
</section>

<section id="edu">
    <div class="content">
    <h2>Education Qualification
    </h2>
    <table boder="1" cellpadding="10" cellspacing="5" width="90%" class="txtalign">
<tr>
    <th>
        YEAR
    </th>
    <th>
        QUALIFICATION
    </th>
    <th>
        INSTITUTION NAME
    </th>
    <th>
        PERCENTAGE
    </th>
</tr>

<tr>
<td>
    2023-2026
</td>
<td>
    BSC IT
</td>
<td>
    Government Arts and SCience College,Valaprai
</td>
<td>
    77%
</td>
</tr>

<tr>
    <td>
        2021-2023
    </td>
    <td>
        HSC
    </td>
    <td>
         THE LITTLE FLOWER HIGHER SECONDARY SCHOOL,SALEM 
    </td>
    <td>
        80%
    </td>
    </tr>
    <tr>
    <td>
        2019-2020
    </td>
    <td>
        SSLC
    </td>
    <td>
        THE LITTLE FLOWER MATRICULATION SCHOOL,SALEM 
    </td>
    <td>
        82%
    </td>
    </tr>

    </table>
    
</div>
</section>


<section id="interest">
    <div class="content">
    <h2>Area of Interest
    </h2>
    <ul>
        <li>Quantum Computing</li>
        <li>Machine Learning</li>
        <li>Full Stack Web Development</li>
        <li>Networks Security</li>
        <li>Cloud Computing</li>
        <li>Data Cloud</li>
        <li>BlockChain Technology</li>
        </ul>
</div>
</section>

<section id="contact">
    <div class="content">
    <h2>Contact
    </h2>
      22/38-N,
        <br>DR.RATHINAM STREET,
        <br>ARISIPALAYAM,
        <br>SALEM-636009
    </div>
</section>

<section id="download">
    <div class="content">
    <h2>Download CV
    </h2>
       <a href="./Cer11.pdf" class="download-button" target="_blank">Download CV</a>
    </div>
</section>

    </body>
</html>



Syle.csss file
a{
    color: rgb(158, 103, 31);
}
a:hover{
    color: rgb(231, 38, 38);
}

    
     *{
        margin: 0;
        /* padding: 2rem 0; */
        font-family: Cambria, Cochin, Georgia, Times, 'Times New Roman', serif;
        font: size 10px;
        box-sizing: border-box;
        
     }

     header{
        background-color: #333;
        color: white;
        position: relative;
        text-align: center;
        padding: 2rem 0;

     }
     .content h1{
        font-size: 2.5rem;
     
     }

     .propic{
            width: 100px;
            height: 100px;
            position: absolute;
            border-radius: 75%;
            top: 75px;
            left: 100px;
            object-fit: cover;
     }

     nav{
       
        background-color: beige;
        text-align: center;
        color: #333;
       
     }
nav ul{
    list-style:none;
    padding: 0px;
}

nav ul li{

    display:inline;
    margin: 0 20px;
    
}
